"""
Invoice AI Extractor v3
=======================
App Streamlit qui extrait automatiquement les données de factures PDF
via l'API Mistral AI (modèle Pixtral, multimodal).

Modes : Single (1 facture en détail) ou Batch (plusieurs factures consolidées).

v3 changelog :
- Split SYSTEM_PROMPT / EXTRACTION_PROMPT (séparation rôle/tâche)
- Ajout d'un few-shot example complet (facture BTP avec acompte + multi-TVA)
- Structure XML du prompt (<schema>, <rules>, <example>, <verification_finale>)
- temperature=0 pour résultats déterministes
- Types harmonisés dans le schéma
- Section de vérification comptable (HT+TVA=TTC, TTC-acomptes=net à payer)

Auteur : Kerem Keles
"""

import os
import json
import base64
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
from mistralai import Mistral

# ---------- CONFIGURATION ----------

# Charge la clé API depuis .env (local) ou st.secrets (Streamlit Cloud)
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Fallback vers st.secrets pour le déploiement Streamlit Cloud
if not MISTRAL_API_KEY:
    try:
        MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

if not MISTRAL_API_KEY:
    st.error("⚠️ Clé API Mistral introuvable. Configure-la dans .env ou dans les Secrets Streamlit.")
    st.stop()

client = Mistral(api_key=MISTRAL_API_KEY)


# ---------- PROMPTS D'EXTRACTION ----------

SYSTEM_PROMPT = """Tu es un expert-comptable spécialisé dans l'extraction de données de factures françaises et européennes.

Ton seul et unique rôle est de transformer une image de facture en un objet JSON STRICT respectant un schéma défini.

Tu ne dois JAMAIS :
- Inventer une information absente de la facture
- Ajouter du texte explicatif autour du JSON
- Utiliser du markdown (pas de ```json, pas de ```)
- Confondre un acompte avec une remise
- Renvoyer un montant au format string si le schéma demande un number

Tu dois TOUJOURS :
- Convertir les montants français (1 234,56 €) en nombres décimaux (1234.56)
- Convertir toutes les dates au format ISO YYYY-MM-DD
- Mettre `null` pour toute information absente
- Renvoyer un JSON syntaxiquement valide
"""


FEWSHOT_EXAMPLE = """<example>
<facture_description>
Une facture de l'entreprise "BATI-PRO SARL" (SIRET 12345678900012) émise le 15/03/2024,
numéro F-2024-0142, adressée au client "Restaurant Le Bistrot" pour des travaux de ravalement.
La facture comporte 2 lignes :
- Ligne 1 : Ravalement façade nord, 80 m² à 45€/m² HT, TVA 10% (travaux rénovation)
- Ligne 2 : Forfait nettoyage chantier, 1 forfait à 200€ HT, TVA 20%
Un acompte de 1500€ a déjà été versé le 01/03/2024.
Total HT = 3800€, TVA = 360€ (10%) + 40€ (20%) = 400€, TTC = 4200€, Net à payer = 2700€.
</facture_description>

<expected_json>
{
  "numero_facture": "F-2024-0142",
  "date_emission": "2024-03-15",
  "date_echeance": null,
  "date_livraison": null,
  "reference_commande": null,
  "reference_devis": null,
  "devise": "EUR",
  "fournisseur": {
    "nom": "BATI-PRO SARL",
    "adresse": "12 rue des Artisans, 76600 Le Havre",
    "siret": "12345678900012",
    "tva_intracom": null,
    "iban": null,
    "bic": null,
    "telephone": null,
    "email": null
  },
  "client": {
    "nom": "Restaurant Le Bistrot",
    "contact": null,
    "adresse": "5 place Gambetta, 76600 Le Havre",
    "siret": null,
    "tva_intracom": null
  },
  "lignes": [
    {
      "reference": null,
      "designation": "Ravalement façade nord",
      "quantite": 80,
      "unite": "m²",
      "prix_unitaire_ht": 45.00,
      "remise_pourcentage": null,
      "taux_tva": 10,
      "total_ht": 3600.00
    },
    {
      "reference": null,
      "designation": "Forfait nettoyage chantier",
      "quantite": 1,
      "unite": "forfait",
      "prix_unitaire_ht": 200.00,
      "remise_pourcentage": null,
      "taux_tva": 20,
      "total_ht": 200.00
    }
  ],
  "ventilation_tva": [
    {"base_ht": 3600.00, "taux": 10, "montant_tva": 360.00},
    {"base_ht": 200.00, "taux": 20, "montant_tva": 40.00}
  ],
  "totaux": {
    "total_ht": 3800.00,
    "total_tva": 400.00,
    "total_ttc": 4200.00,
    "acomptes_verses": [
      {"date": "2024-03-01", "montant": 1500.00, "description": "Acompte versé à la commande"}
    ],
    "net_a_payer": 2700.00
  },
  "mode_paiement": "Virement bancaire",
  "conditions_paiement": "30 jours fin de mois",
  "informations_complementaires": {
    "chantier": "Façade restaurant Le Bistrot",
    "garanties": null,
    "mentions_legales_specifiques": null,
    "autres": null
  }
}
</expected_json>
</example>"""


EXTRACTION_PROMPT = """Analyse l'image de facture ci-jointe et produis un JSON respectant exactement le schéma indiqué.

<schema>
{
  "numero_facture": string,
  "date_emission": "YYYY-MM-DD",
  "date_echeance": "YYYY-MM-DD" | null,
  "date_livraison": "YYYY-MM-DD" | null,
  "reference_commande": string | null,
  "reference_devis": string | null,
  "devise": "EUR" | "USD" | "GBP" | autre code ISO,

  "fournisseur": {
    "nom": string,
    "adresse": string,
    "siret": string | null,
    "tva_intracom": string | null,
    "iban": string | null,
    "bic": string | null,
    "telephone": string | null,
    "email": string | null
  },

  "client": {
    "nom": string,
    "contact": string | null,
    "adresse": string,
    "siret": string | null,
    "tva_intracom": string | null
  },

  "lignes": [
    {
      "reference": string | null,
      "designation": string,
      "quantite": number,
      "unite": string | null,
      "prix_unitaire_ht": number,
      "remise_pourcentage": number | null,
      "taux_tva": number | null,
      "total_ht": number
    }
  ],

  "ventilation_tva": [
    {"base_ht": number, "taux": number, "montant_tva": number}
  ],

  "totaux": {
    "total_ht": number,
    "total_tva": number,
    "total_ttc": number,
    "acomptes_verses": [
      {"date": "YYYY-MM-DD" | null, "montant": number, "description": string | null}
    ],
    "net_a_payer": number
  },

  "mode_paiement": string | null,
  "conditions_paiement": string | null,

  "informations_complementaires": {
    "chantier": string | null,
    "garanties": string | null,
    "mentions_legales_specifiques": string | null,
    "autres": string | null
  }
}
</schema>

<rules>
1. CAPTURE TOUTES les lignes du tableau, même sur plusieurs pages. Compte-les avant de produire le JSON.
2. ACOMPTES : toute mention "acompte versé", "déjà payé", "déduction" va dans `acomptes_verses`. NE JAMAIS confondre avec une remise.
3. REMISES : si une ligne a une remise (-5%, -10€), capture-la dans `remise_pourcentage` de la ligne.
4. MULTI-TVA : un objet par taux distinct dans `ventilation_tva`.
5. NET À PAYER : si mentionné explicitement et différent du TTC (à cause d'acomptes), capture-le. Sinon = total_ttc.
6. DEVISE : code ISO (EUR, USD, GBP).
7. DATES : format YYYY-MM-DD strict.
8. MONTANTS : nombres décimaux uniquement. "1 234,56" devient 1234.56. JAMAIS de string pour un montant.
9. ABSENT = null. Ne jamais inventer.
10. Réponse = JSON pur uniquement. Aucun texte avant ou après. Aucun markdown.
</rules>

<example_de_reference>
Voici un exemple complet du format attendu pour une facture BTP type :
""" + FEWSHOT_EXAMPLE + """
</example_de_reference>

<verification_finale>
Avant de produire ta réponse, vérifie mentalement :
- As-tu capturé TOUTES les lignes ?
- Tous les acomptes sont-ils dans `acomptes_verses` (pas dans les remises) ?
- Les TVA sont-elles correctement ventilées par taux ?
- Tous les montants sont-ils des nombres (pas des strings) ?
- Total HT + Total TVA = Total TTC ? (si non, relis la facture)
- Total TTC - somme(acomptes) = Net à payer ?
</verification_finale>

Produis maintenant le JSON, et UNIQUEMENT le JSON.
"""


# ---------- HELPERS ----------

def pdf_to_images_base64(pdf_bytes: bytes, dpi: int = 150) -> list[str]:
    """Convertit un PDF en liste d'images PNG encodées en base64."""
    images_base64 = []
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page in pdf_doc:
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix)
        img_bytes = pixmap.tobytes("png")
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        images_base64.append(img_base64)

    pdf_doc.close()
    return images_base64


def extract_invoice_data(pdf_bytes: bytes) -> dict:
    """
    Extrait les données structurées d'une facture PDF via l'API Mistral.

    Args:
        pdf_bytes: Le contenu brut du PDF

    Returns:
        dict: Les données extraites de la facture

    Raises:
        Exception: Si l'extraction échoue
    """
    images_b64 = pdf_to_images_base64(pdf_bytes)

    # Construction du message user : images + instruction
    user_content = []
    for img_b64 in images_b64:
        user_content.append({
            "type": "image_url",
            "image_url": f"data:image/png;base64,{img_b64}",
        })
    user_content.append({
        "type": "text",
        "text": EXTRACTION_PROMPT,
    })

    response = client.chat.complete(
        model="pixtral-12b-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    raw_text = response.choices[0].message.content.strip()

    # Nettoyage défensif des backticks markdown (ceinture + bretelles)
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    return json.loads(raw_text)


def invoice_to_excel(data: dict) -> bytes:
    """Convertit les données extraites en fichier Excel multi-feuilles (mode Single)."""
    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        infos = {
            "Champ": [
                "Numéro de facture", "Date d'émission", "Date d'échéance",
                "Date de livraison", "Référence commande", "Référence devis",
                "Devise", "Mode de paiement", "Conditions de paiement",
            ],
            "Valeur": [
                data.get("numero_facture", ""),
                data.get("date_emission", ""),
                data.get("date_echeance", ""),
                data.get("date_livraison", ""),
                data.get("reference_commande", ""),
                data.get("reference_devis", ""),
                data.get("devise", ""),
                data.get("mode_paiement", ""),
                data.get("conditions_paiement", ""),
            ],
        }
        pd.DataFrame(infos).to_excel(writer, sheet_name="Infos", index=False)

        fournisseur = data.get("fournisseur", {})
        df_fourn = pd.DataFrame({
            "Champ": ["Nom", "Adresse", "SIRET", "TVA Intracom", "IBAN", "BIC", "Téléphone", "Email"],
            "Valeur": [
                fournisseur.get("nom", ""), fournisseur.get("adresse", ""),
                fournisseur.get("siret", ""), fournisseur.get("tva_intracom", ""),
                fournisseur.get("iban", ""), fournisseur.get("bic", ""),
                fournisseur.get("telephone", ""), fournisseur.get("email", ""),
            ],
        })
        df_fourn.to_excel(writer, sheet_name="Fournisseur", index=False)

        client_info = data.get("client", {})
        df_client = pd.DataFrame({
            "Champ": ["Nom", "Contact", "Adresse", "SIRET", "TVA Intracom"],
            "Valeur": [
                client_info.get("nom", ""), client_info.get("contact", ""),
                client_info.get("adresse", ""), client_info.get("siret", ""),
                client_info.get("tva_intracom", ""),
            ],
        })
        df_client.to_excel(writer, sheet_name="Client", index=False)

        lignes = data.get("lignes", [])
        if lignes:
            pd.DataFrame(lignes).to_excel(writer, sheet_name="Lignes", index=False)

        ventilation = data.get("ventilation_tva", [])
        if ventilation:
            pd.DataFrame(ventilation).to_excel(writer, sheet_name="Ventilation TVA", index=False)

        acomptes = data.get("totaux", {}).get("acomptes_verses", [])
        if acomptes:
            pd.DataFrame(acomptes).to_excel(writer, sheet_name="Acomptes", index=False)

        totaux = data.get("totaux", {})
        df_totaux = pd.DataFrame({
            "Champ": ["Total HT", "Total TVA", "Total TTC", "Net à payer"],
            "Valeur": [
                totaux.get("total_ht", 0), totaux.get("total_tva", 0),
                totaux.get("total_ttc", 0), totaux.get("net_a_payer", 0),
            ],
        })
        df_totaux.to_excel(writer, sheet_name="Totaux", index=False)

    output.seek(0)
    return output.getvalue()


def invoice_to_csv(data: dict) -> str:
    """Convertit les données extraites en CSV à plat (1 ligne par facture)."""
    fournisseur = data.get("fournisseur", {})
    client_info = data.get("client", {})
    totaux = data.get("totaux", {})

    acomptes = totaux.get("acomptes_verses", [])
    total_acomptes = sum(a.get("montant", 0) for a in acomptes) if acomptes else 0

    flat = {
        "numero_facture": data.get("numero_facture", ""),
        "date_emission": data.get("date_emission", ""),
        "date_echeance": data.get("date_echeance", ""),
        "devise": data.get("devise", ""),
        "fournisseur_nom": fournisseur.get("nom", ""),
        "fournisseur_siret": fournisseur.get("siret", ""),
        "fournisseur_tva": fournisseur.get("tva_intracom", ""),
        "client_nom": client_info.get("nom", ""),
        "client_siret": client_info.get("siret", ""),
        "total_ht": totaux.get("total_ht", 0),
        "total_tva": totaux.get("total_tva", 0),
        "total_ttc": totaux.get("total_ttc", 0),
        "total_acomptes": total_acomptes,
        "net_a_payer": totaux.get("net_a_payer", 0),
        "iban": fournisseur.get("iban", ""),
        "mode_paiement": data.get("mode_paiement", ""),
    }

    df = pd.DataFrame([flat])
    return df.to_csv(index=False, sep=';', encoding='utf-8-sig')


def invoice_to_flat_dict(data: dict, filename: str) -> dict:
    """Aplatit une facture en dict flat (pour batch consolidation)."""
    fournisseur = data.get("fournisseur", {})
    client_info = data.get("client", {})
    totaux = data.get("totaux", {})

    acomptes = totaux.get("acomptes_verses", [])
    total_acomptes = sum(a.get("montant", 0) for a in acomptes) if acomptes else 0
    nb_lignes = len(data.get("lignes", []))

    return {
        "fichier_source": filename,
        "numero_facture": data.get("numero_facture", ""),
        "date_emission": data.get("date_emission", ""),
        "date_echeance": data.get("date_echeance", ""),
        "devise": data.get("devise", "EUR"),
        "fournisseur_nom": fournisseur.get("nom", ""),
        "fournisseur_siret": fournisseur.get("siret", ""),
        "fournisseur_tva": fournisseur.get("tva_intracom", ""),
        "client_nom": client_info.get("nom", ""),
        "client_siret": client_info.get("siret", ""),
        "nb_lignes": nb_lignes,
        "total_ht": totaux.get("total_ht", 0),
        "total_tva": totaux.get("total_tva", 0),
        "total_ttc": totaux.get("total_ttc", 0),
        "total_acomptes": total_acomptes,
        "net_a_payer": totaux.get("net_a_payer", 0),
        "iban": fournisseur.get("iban", ""),
        "mode_paiement": data.get("mode_paiement", ""),
    }


def batch_to_excel(batch_results: list[dict], errors: list[dict]) -> bytes:
    """Génère un fichier Excel consolidé pour le batch."""
    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Feuille 1 : Récapitulatif (1 ligne par facture)
        if batch_results:
            df_recap = pd.DataFrame(batch_results)
            df_recap.to_excel(writer, sheet_name="Récapitulatif", index=False)

        # Feuille 2 : Erreurs (s'il y en a)
        if errors:
            df_errors = pd.DataFrame(errors)
            df_errors.to_excel(writer, sheet_name="Erreurs", index=False)

        # Feuille 3 : Statistiques globales
        if batch_results:
            df_results = pd.DataFrame(batch_results)
            stats_data = {
                "Indicateur": [
                    "Nombre de factures traitées",
                    "Nombre d'erreurs",
                    "Total HT cumulé",
                    "Total TVA cumulé",
                    "Total TTC cumulé",
                    "Net à payer cumulé",
                    "Facture la plus élevée (TTC)",
                    "Facture la plus basse (TTC)",
                    "Moyenne TTC",
                ],
                "Valeur": [
                    len(batch_results),
                    len(errors),
                    f"{df_results['total_ht'].sum():,.2f}",
                    f"{df_results['total_tva'].sum():,.2f}",
                    f"{df_results['total_ttc'].sum():,.2f}",
                    f"{df_results['net_a_payer'].sum():,.2f}",
                    f"{df_results['total_ttc'].max():,.2f}",
                    f"{df_results['total_ttc'].min():,.2f}",
                    f"{df_results['total_ttc'].mean():,.2f}",
                ],
            }
            pd.DataFrame(stats_data).to_excel(writer, sheet_name="Statistiques", index=False)

    output.seek(0)
    return output.getvalue()


# ---------- INTERFACE STREAMLIT ----------

st.set_page_config(
    page_title="Invoice AI Extractor",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Invoice AI Extractor")
st.markdown(
    "Extraction automatique de données de factures PDF par IA. "
    "Mode **Single** pour une facture en détail, mode **Batch** pour traiter plusieurs factures d'un coup."
)

# Sélecteur de mode
mode = st.radio(
    "Mode d'utilisation",
    options=["📄 Single (1 facture)", "📚 Batch (plusieurs factures)"],
    horizontal=True,
)

st.divider()

# ============================================================
# MODE SINGLE — Une facture à la fois
# ============================================================

if mode == "📄 Single (1 facture)":
    uploaded_file = st.file_uploader(
        "Choisissez une facture PDF",
        type=["pdf"],
        help="Glissez-déposez ou cliquez pour sélectionner un fichier PDF",
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📑 Facture")
            st.write(f"**Nom du fichier :** {uploaded_file.name}")
            st.write(f"**Taille :** {uploaded_file.size / 1024:.1f} KB")

        with col2:
            st.subheader("🤖 Extraction IA")

            if st.button("🚀 Extraire les données", type="primary"):
                with st.spinner("L'IA analyse la facture..."):
                    try:
                        pdf_bytes = uploaded_file.read()
                        data = extract_invoice_data(pdf_bytes)
                        st.success("✅ Extraction réussie !")
                        st.session_state["extracted_data"] = data
                    except json.JSONDecodeError as e:
                        st.error(f"❌ Erreur de parsing JSON : {e}")
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'extraction : {e}")
                        st.exception(e)

    # Affichage des données extraites (mode single)
    if "extracted_data" in st.session_state:
        st.divider()
        st.subheader("📊 Données extraites")

        data = st.session_state["extracted_data"]
        devise = data.get("devise", "EUR")

        metrics_cols = st.columns(4)
        with metrics_cols[0]:
            st.metric("N° Facture", data.get("numero_facture", "N/A"))
        with metrics_cols[1]:
            st.metric("Date émission", data.get("date_emission", "N/A"))
        with metrics_cols[2]:
            total_ttc = data.get("totaux", {}).get("total_ttc", 0)
            st.metric(f"Total TTC ({devise})", f"{total_ttc:,.2f}")
        with metrics_cols[3]:
            net = data.get("totaux", {}).get("net_a_payer", 0)
            st.metric(f"Net à payer ({devise})", f"{net:,.2f}")

        detail_cols = st.columns(2)
        with detail_cols[0]:
            st.markdown("### 🏢 Fournisseur")
            fournisseur = data.get("fournisseur", {})
            st.write(f"**Nom :** {fournisseur.get('nom', 'N/A')}")
            st.write(f"**Adresse :** {fournisseur.get('adresse', 'N/A')}")
            st.write(f"**SIRET :** {fournisseur.get('siret', 'N/A')}")
            st.write(f"**TVA :** {fournisseur.get('tva_intracom', 'N/A')}")
            if fournisseur.get('iban'):
                st.write(f"**IBAN :** {fournisseur.get('iban')}")

        with detail_cols[1]:
            st.markdown("### 🧾 Client")
            client_info = data.get("client", {})
            st.write(f"**Nom :** {client_info.get('nom', 'N/A')}")
            if client_info.get('contact'):
                st.write(f"**Contact :** {client_info.get('contact')}")
            st.write(f"**Adresse :** {client_info.get('adresse', 'N/A')}")
            st.write(f"**SIRET :** {client_info.get('siret', 'N/A')}")

        st.markdown("### 📋 Lignes de facture")
        lignes = data.get("lignes", [])
        if lignes:
            st.dataframe(lignes, use_container_width=True)
            st.caption(f"📊 {len(lignes)} ligne(s) extraite(s)")
        else:
            st.info("Aucune ligne détectée.")

        ventilation = data.get("ventilation_tva", [])
        if ventilation and len(ventilation) > 0:
            st.markdown("### 💶 Ventilation TVA")
            st.dataframe(ventilation, use_container_width=True)

        acomptes = data.get("totaux", {}).get("acomptes_verses", [])
        if acomptes and len(acomptes) > 0:
            st.markdown("### 💸 Acomptes versés")
            st.dataframe(acomptes, use_container_width=True)
            total_acomptes = sum(a.get("montant", 0) for a in acomptes)
            st.caption(f"💰 Total des acomptes : {total_acomptes:,.2f} {devise}")

        st.markdown("### 💰 Récapitulatif financier")
        totaux = data.get("totaux", {})
        recap_cols = st.columns(4)
        with recap_cols[0]:
            st.metric("Total HT", f"{totaux.get('total_ht', 0):,.2f}")
        with recap_cols[1]:
            st.metric("Total TVA", f"{totaux.get('total_tva', 0):,.2f}")
        with recap_cols[2]:
            st.metric("Total TTC", f"{totaux.get('total_ttc', 0):,.2f}")
        with recap_cols[3]:
            st.metric("NET À PAYER", f"{totaux.get('net_a_payer', 0):,.2f}")

        info_comp = data.get("informations_complementaires", {})
        if info_comp and any(info_comp.values()):
            with st.expander("ℹ️ Informations complémentaires"):
                for key, value in info_comp.items():
                    if value:
                        st.write(f"**{key.replace('_', ' ').title()} :** {value}")

        st.markdown("### 💾 Export")
        export_cols = st.columns(3)
        invoice_num = data.get('numero_facture', 'extraction').replace('/', '-')

        with export_cols[0]:
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 JSON",
                data=json_str,
                file_name=f"{invoice_num}.json",
                mime="application/json",
                use_container_width=True,
            )
        with export_cols[1]:
            excel_bytes = invoice_to_excel(data)
            st.download_button(
                label="📊 Excel",
                data=excel_bytes,
                file_name=f"{invoice_num}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with export_cols[2]:
            csv_str = invoice_to_csv(data)
            st.download_button(
                label="📋 CSV",
                data=csv_str,
                file_name=f"{invoice_num}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with st.expander("🔍 Voir le JSON brut"):
            st.code(json_str, language="json")


# ============================================================
# MODE BATCH — Plusieurs factures consolidées
# ============================================================

else:  # mode == "📚 Batch (plusieurs factures)"

    st.markdown("""
    📚 **Mode Batch** : uploadez plusieurs factures PDF en une seule fois.
    L'IA traite chaque facture et vous obtenez :
    - Un tableau récapitulatif (1 ligne par facture)
    - Un fichier Excel consolidé prêt pour la compta
    - Statistiques globales (totaux cumulés, moyennes, etc.)
    """)

    uploaded_files = st.file_uploader(
        "Choisissez plusieurs factures PDF",
        type=["pdf"],
        accept_multiple_files=True,
        help="Sélectionnez ou glissez plusieurs PDF à la fois",
    )

    if uploaded_files:
        st.success(f"✅ **{len(uploaded_files)} fichier(s)** prêts à traiter")

        # Affichage de la liste des fichiers
        with st.expander(f"📋 Voir la liste des {len(uploaded_files)} fichier(s)"):
            for i, f in enumerate(uploaded_files, 1):
                st.write(f"{i}. {f.name} ({f.size / 1024:.1f} KB)")

        # Bouton lancement traitement
        if st.button("🚀 Lancer le traitement batch", type="primary"):

            # Initialisation des conteneurs de résultats
            batch_results = []
            errors = []

            # Barre de progression et zone de status
            progress_bar = st.progress(0, text="Préparation...")
            status_container = st.container()

            # Traitement de chaque facture
            for i, uploaded_file in enumerate(uploaded_files):
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(
                    progress,
                    text=f"📄 Traitement {i+1}/{len(uploaded_files)} : {uploaded_file.name}"
                )

                try:
                    # Lecture du fichier
                    pdf_bytes = uploaded_file.read()

                    # Extraction
                    data = extract_invoice_data(pdf_bytes)

                    # Aplatissement pour la consolidation
                    flat = invoice_to_flat_dict(data, uploaded_file.name)
                    batch_results.append(flat)

                    with status_container:
                        st.success(
                            f"✅ {uploaded_file.name} → "
                            f"Facture {flat['numero_facture']} - "
                            f"{flat['total_ttc']:,.2f} {flat['devise']}"
                        )

                except Exception as e:
                    errors.append({
                        "fichier": uploaded_file.name,
                        "erreur": str(e),
                    })
                    with status_container:
                        st.error(f"❌ {uploaded_file.name} → Erreur : {str(e)[:100]}")

            progress_bar.progress(1.0, text="✅ Traitement terminé !")

            # Stockage en session pour réafficher
            st.session_state["batch_results"] = batch_results
            st.session_state["batch_errors"] = errors

    # Affichage des résultats batch
    if "batch_results" in st.session_state and st.session_state["batch_results"]:
        st.divider()
        st.subheader("📊 Résultats du traitement batch")

        results = st.session_state["batch_results"]
        errors = st.session_state.get("batch_errors", [])

        # KPIs globaux
        df_results = pd.DataFrame(results)
        kpi_cols = st.columns(4)
        with kpi_cols[0]:
            st.metric("✅ Factures OK", len(results))
        with kpi_cols[1]:
            st.metric("❌ Erreurs", len(errors))
        with kpi_cols[2]:
            st.metric("💰 Total HT", f"{df_results['total_ht'].sum():,.2f} €")
        with kpi_cols[3]:
            st.metric("💵 Total TTC", f"{df_results['total_ttc'].sum():,.2f} €")

        # Tableau récapitulatif
        st.markdown("### 📋 Tableau récapitulatif")
        st.dataframe(df_results, use_container_width=True)

        # Tableau erreurs s'il y en a
        if errors:
            st.markdown("### ⚠️ Factures en erreur")
            st.dataframe(pd.DataFrame(errors), use_container_width=True)

        # Boutons d'export
        st.markdown("### 💾 Export consolidé")
        export_cols = st.columns(2)

        with export_cols[0]:
            excel_bytes = batch_to_excel(results, errors)
            st.download_button(
                label="📊 Télécharger Excel consolidé",
                data=excel_bytes,
                file_name="batch_factures_consolidees.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                type="primary",
            )

        with export_cols[1]:
            csv_str = df_results.to_csv(index=False, sep=';', encoding='utf-8-sig')
            st.download_button(
                label="📋 Télécharger CSV consolidé",
                data=csv_str,
                file_name="batch_factures_consolidees.csv",
                mime="text/csv",
                use_container_width=True,
            )

# Footer
st.divider()
st.caption(
    "Construit avec Streamlit + Mistral AI (Pixtral) · "
    "Par Kerem Keles · [GitHub](https://github.com/keleskerem11)"
)
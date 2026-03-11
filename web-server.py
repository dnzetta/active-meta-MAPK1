import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from joblib import load
from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (
    GraphConv,
    NNConv,
    global_mean_pool
)
from torch_geometric.utils import to_dense_batch

# --- Unified Styling ---
st.markdown("""
<style>
/* ===== PAGE BACKGROUND ===== */
.stApp {
background-color: #eef6fa !important;
font-family: 'sans-serif' !important;
}
            
/* ===== GLOBAL TEXT ===== */
p, span, div, label, h1, h2, h3, h4, h5 {
    color: #002244 !important;
}
            
/* ===== AUTHOR SECTION ===== */
.author {
    background-color: #cce0ff !important;
    color: #003366 !important;
    font-style: italic;
    font-size: 16px;
    text-align: center;
    padding: 15px;
    border-radius: 10px;
    margin-top: 30px;
}

/* ===== TABS ===== */
[data-testid="stTabs"] div[role="tablist"] div[role="tab"] button,
[data-testid="stTabs"] div[role="tablist"] div[role="tab"] button span,
[data-testid="stTabs"] div[role="tablist"] div[role="tab"] button div {
    color: #002244 !important;
    background-color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    opacity: 1 !important;
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stTabs"] div[data-baseweb="tab-panel"] * {
    color: #002244 !important;
}

/* ===== INPUT BOXES ===== */
div[data-baseweb="input"] input {
    background-color: #ffffff !important;
    color: #002244 !important;
    border: 1px solid #4da6ff !important;
    border-radius: 8px !important;
    padding: 6px 10px !important;
}
div[data-baseweb="input"] input::placeholder {
    color: #666666 !important;
}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] section {
    background-color: #ffffff !important;
    color: #002244 !important;
    border: 1px dashed #4da6ff !important;
    border-radius: 8px !important;
    padding: 10px !important;
}
[data-testid="stFileUploader"] button {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    border: 1px solid #000000 !important;
    padding: 8px 16px !important;
}
[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploader"] button:focus {
    background-color: #f0f0f0 !important;
    color: #000000 !important;
}

/* ===== ALL BUTTONS ===== */
div.stButton > button {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    border: 1px solid #000000 !important;
    padding: 8px 16px !important;
    }
/* Hover/focus state */ 
div.stButton > button:hover, 
div.stButton > button:focus { 
    background-color: #f0f0f0 !important; /* 
    Slight gray on hover */ 
    color: #000000 !important; 
    }
            
   
</style>
""", unsafe_allow_html=True)

# =========================
# Page config
# =========================
st.set_page_config(page_title="MAPK1 Inhibitor Screening", layout="centered")
st.title("🔬 MAPK1 Inhibitor Screening")

# --- Introduction Section ---
st.markdown("#### 🧠 Active Meta-Deep Learning Framework")
st.info("""
This platform predicts mitogen-activated protein kinase 1 **(MAPK1) inhibitors** by integrating molecular descriptors and graph-based features through an active meta-deep learning framework.
""")

# --- Model Architecture Section ---
st.markdown("#### Run Prediction")
st.markdown("""
* **Base Models:** Attention, convolutional neural network (CNN), graph convolutional network (GCN), and graph neural network (GNN)-attention.
* **Meta Model:** Meta-attention model.
* **Active Learning Strategy:** Optimized during training using **entropy-based sampling** to select the most informative data points.
* **Features:** SMILES-based descriptors and molecular graphs.
""")

# --- Prediction Input (Placeholder) ---
st.markdown("#### Run Prediction")
# =========================
# Load models & scalers
# =========================
class GCNNClassifier(nn.Module):
    def __init__(self, node_dim, hidden_dim=64, num_layers=3):
        super(GCNNClassifier, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(node_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))

        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x).squeeze(1)
    
class GMPNNClassifier(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_heads=4):
        super(GMPNNClassifier, self).__init__()
        self.edge_net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim * hidden_dim)
        )
        self.nnconv = NNConv(
            in_channels=node_dim,
            out_channels=hidden_dim,
            nn=self.edge_net,
            aggr='mean'
        )
        
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Message passing
        x = self.nnconv(x, edge_index, edge_attr)
        x = F.relu(x)

        # Convert to dense batch: (batch_size, max_num_nodes, hidden_dim)
        x_dense, mask = to_dense_batch(x, batch)  # mask is (batch_size, max_num_nodes)

        # Apply MultiheadAttention (query, key, value are all x)
        attn_output, _ = self.multihead_attn(x_dense, x_dense, x_dense, key_padding_mask=~mask)

        # Aggregate the output: mean over node dimension (masked)
        attn_output[~mask] = 0  # mask out padded nodes
        graph_embeddings = attn_output.sum(dim=1) / mask.sum(dim=1, keepdim=True)  # (batch_size, hidden_dim)

        # Final MLP
        x = F.relu(self.lin1(graph_embeddings))
        return self.lin2(x).squeeze(1)

@st.cache_resource
def load_resources():
    # ---- Keras models ----
    cnn = load_model("cnn_model.keras")
    att = load_model("attention_model.keras")
    meta = load_model("meta_attention_model.keras")

    # ---- PyTorch models ----
    node_dim = 8 # As defined in atom_features() function
    edge_dim = 4

    gcn = GCNNClassifier(node_dim)
    gcn.load_state_dict(torch.load("gcn_model.keras", map_location="cpu"))
    gcn.eval()

    gat = GMPNNClassifier(node_dim, edge_dim)
    gat.load_state_dict(torch.load("gnn_attention_model.keras", map_location="cpu"))
    gat.eval()

    return {
        "scaler_cnn": load("scaler_cnn.joblib"),
        "scaler_att": load("scaler_attention.joblib"),
        "scaler_meta": load("scaler_meta_attention.joblib"),

        "cnn": cnn,
        "att": att,
        "meta": meta,
        "gcn": gcn,
        "gat": gat,
    }


res = load_resources()

# =========================
# Descriptor extractor
# =========================
def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    funcs = [
        Descriptors.MolWt,
        Descriptors.MolLogP,
        Descriptors.NumHDonors,
        Descriptors.NumHAcceptors,
        rdMolDescriptors.CalcTPSA,
        Descriptors.NumRotatableBonds,
        Descriptors.NumAromaticRings,
        rdMolDescriptors.CalcNumAromaticCarbocycles,
        rdMolDescriptors.CalcNumAromaticHeterocycles,
        rdMolDescriptors.CalcNumSaturatedRings,
        rdMolDescriptors.CalcNumHeteroatoms,
        rdMolDescriptors.CalcNumRings,
        rdMolDescriptors.CalcNumHeavyAtoms,
        rdMolDescriptors.CalcNumAliphaticRings,
        rdMolDescriptors.CalcNumAliphaticCarbocycles,
        rdMolDescriptors.CalcNumAliphaticHeterocycles,
        Descriptors.NumValenceElectrons,
        rdMolDescriptors.CalcNumSpiroAtoms,
        rdMolDescriptors.CalcNumHeterocycles,
        rdMolDescriptors.CalcNumAmideBonds,
    ]

    values = [f(mol) for f in funcs]
    return np.array(values, dtype=float).reshape(1, -1)

# =========================
# Graph encoder (Keras)
# =========================
def atom_features(atom):
    return torch.tensor([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetChiralTag()),
        atom.GetTotalNumHs(),
        int(atom.GetHybridization()),
        atom.GetIsAromatic(),
        atom.GetMass(),
    ], dtype=torch.float)

def bond_features(bond):
    return torch.tensor([
        float(bond.GetBondTypeAsDouble()),
        bond.IsInRing(),
        int(bond.GetStereo()),
        bond.GetIsConjugated(),
    ], dtype=torch.float)

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])

    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        feat = bond_features(bond)

        edge_index += [[i, j], [j, i]]
        edge_attr += [feat, feat]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=torch.zeros(x.size(0), dtype=torch.long)
    )

    return data


# =========================
# Full inference pipeline
# =========================
def predict_full_system(smiles):
    # Descriptor branch
    x_desc = smiles_to_descriptors(smiles)
    if x_desc is None:
        return None

    x_cnn = res["scaler_cnn"].transform(x_desc)
    x_att = res["scaler_att"].transform(x_desc)
    cnn = res["cnn"].predict(x_cnn, verbose=0)
    att = res["att"].predict(x_att, verbose=0)
    
    # Graph branch
    graph = smiles_to_graph(smiles)
    if graph is None:
        return None

    with torch.no_grad():
        gcn = torch.sigmoid(res["gcn"](graph)).cpu().numpy().reshape(1, 1)
        gat = torch.sigmoid(res["gat"](graph)).cpu().numpy().reshape(1, 1)


    # Stack
    stack = np.hstack([cnn, att, gcn, gat])

    # Meta scaling
    stack_scaled = res["scaler_meta"].transform(stack)

    # Meta attention
    final = res["meta"].predict(stack_scaled, verbose=0)

    return final, stack

# =========================
# UI
# =========================
tab1, tab2 = st.tabs(["Single SMILES", "CSV Batch"])

with tab1:
    smiles_input = st.text_input("🔹 Enter SMILES:")
    if st.button("Predict SMILES"):
        result = predict_full_system(smiles_input)

        if result is None:
            st.error("Invalid SMILES")
        else:
            final, stack = result
            prob = final[0][0]

            st.metric("Final Probability", f"{prob:.4f}")
            st.write("Base model outputs:")
            st.write(f"CNN: {stack[0][0]:.4f}")
            st.write(f"Attention: {stack[0][1]:.4f}")
            st.write(f"GCN: {stack[0][2]:.4f}")
            st.write(f"GAT: {stack[0][3]:.4f}")

            # Textual interpretation
            st.markdown("**Interpretation:**")
            if prob < 0.5:
                st.success("Inactive")
            elif prob == 0.5:
                st.warning("Uncertain")
            else:
                st.error("Active")

            # Heatmap visualization below metrics
            st.write("**Prediction Heatmap:**")
            fig, ax = plt.subplots(figsize=(1.5, 1.5), dpi=500)  # smaller figure
            sns.heatmap(
                [[prob]],
                vmin=0, vmax=1,
                cmap="RdYlGn_r",
                annot=True,
                fmt=".3f",
                cbar=True,
                annot_kws={"size": 5},  # smaller font
                ax=ax
            )

            # Make colorbar tick labels smaller
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=4)

            # Remove axis labels/ticks
            ax.set_xticks([])
            ax.set_yticks([])

            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)  # prevent Streamlit from stretching

            # Explanation just below heatmap
            st.markdown(
                "**Heatmap Interpretation:**  \n"
                "\\< 0.5 → Inactive  \n"
                "= 0.5 → Uncertain  \n"
                "\\> 0.5 → Active"
            )


with tab2:
    uploaded_file = st.file_uploader("📂 Upload CSV with SMILES column", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        preds = []
        for smiles in df["SMILES"]:
            result = predict_full_system(smiles)
            preds.append(result[0][0] if result else None)

        df["Predicted_Probability"] = preds

        df["Inhibitor_Label"] = df["Predicted_Probability"].apply(
            lambda x: "Invalid" if x is None
            else ("Uncertain" if x == 0.5
            else ("Active" if x > 0.5
            else "Inactive"))
        )

        st.dataframe(df)
        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            "predictions.csv",
            "text/csv"
        )

        # Heatmap legend
        st.markdown(
            "**Prediction Interpretation:**  \n"
            "\\< 0.5 → Inactive  \n"
            "= 0.5 → Uncertain  \n"
            "\\> 0.5 → Active"
        )


# =========================
# Footer
# =========================
# --- Spacer before author section ---
st.markdown("<br><br><br>", unsafe_allow_html=True)

# --- Author Section ---
st.markdown("""
<div class="author">
Authors\n
Darlene Nabila Zetta<sup>1</sup> and Tarapong Srisongkram<sup>2*</sup>  

<sup>1</sup>*Graduate School in the Program of Pharmaceutical Sciences, Faculty of Pharmaceutical Sciences, Khon Kaen University, Khon Kaen 40002, Thailand*  
<sup>2</sup>*Division of Pharmaceutical Chemistry, Faculty of Pharmaceutical Sciences, Khon Kaen University, Khon Kaen 40002, Thailand*
</div>
""", unsafe_allow_html=True)

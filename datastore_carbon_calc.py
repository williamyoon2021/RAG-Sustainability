import math
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# === Configuration ===

token_counts = [14e9, 140e9, 350e9, 700e9, 1e12, 1.4e12]

# Datastore + embedding config
CHUNK_SIZE = 256
EMBEDDING_DIM = 768
FLOAT32_SIZE = 4               # Each float32 is 4 bytes
# PQ_COMPRESSED_SIZE = 64        # FAISS PQ-compressed vector size (bytes)
TOKENS_PER_BYTE = 1            # 1 token = 1 byte
METADATA_BYTES_PER_CHUNK = 16  # Metadata size per chunk

# Storage device specs
SSD_SIZE_TB = 0.96
CARBON_INTENSITY = 16.916      # kg CO₂e per provisioned TB

# GPU model info
models = {
    "Llama 3 8B": {"vram_gb": 16.0792161, "gpu_count": 2},
    "Llama 2 7B": {"vram_gb": 13.5017859, "gpu_count": 1},
    "Llama 2 13B": {"vram_gb": 26.001786, "gpu_count": 2}
}
T4_CARBON_PER_UNIT_KG = 41.8  # kg CO₂e per T4 GPU manufacture

# === Helpers ===

def bytes_to_tb(bytes_val):
    """Convert bytes to terabytes (TB)."""
    return bytes_val / 1024**4

def calculate_storage_impact(total_tb):
    """Given TB needed, calculate SSD count, provisioned TB, and carbon impact."""
    num_ssds = math.ceil(total_tb / SSD_SIZE_TB)
    total_utilized_tb = num_ssds * SSD_SIZE_TB
    carbon_kg = total_utilized_tb * CARBON_INTENSITY
    return num_ssds, total_utilized_tb, carbon_kg

# === Main Computation ===

# rows = []

# for total_tokens in token_counts:
#     # Split total tokens into fixed-size chunks
#     num_chunks = int(total_tokens // CHUNK_SIZE)

#     # Base storage: raw text, metadata, embeddings
#     # only include once
#     raw_text_bytes = int(total_tokens * TOKENS_PER_BYTE)

#     metadata_bytes = num_chunks * METADATA_BYTES_PER_CHUNK
#     embedding_uncompressed_bytes = num_chunks * EMBEDDING_DIM * FLOAT32_SIZE
#     embedding_compressed_bytes = num_chunks * PQ_COMPRESSED_SIZE

#     # Total data index size (raw + metadata + embeddings)
#     uncompressed_index_bytes = raw_text_bytes + metadata_bytes + embedding_uncompressed_bytes
#     compressed_index_bytes = raw_text_bytes + metadata_bytes + embedding_compressed_bytes

#     # Double-count raw text by adding it again for retrieval-side storage
#     total_uncompressed_tb = bytes_to_tb(raw_text_bytes + uncompressed_index_bytes)
#     total_compressed_tb = bytes_to_tb(raw_text_bytes + compressed_index_bytes)

#     # add faiss index to both, total size of retrieval indices
#     # mess around quantization
#     # translate bytes to SSD storage
#     # data store size to embodied carbon

#     # Calculate provisioned SSDs and storage carbon
#     uc_ssds, uc_tb_used, uc_carbon = calculate_storage_impact(total_uncompressed_tb)
#     c_ssds, c_tb_used, c_carbon = calculate_storage_impact(total_compressed_tb)

#     # For each model, add GPU carbon impact
#     for model_name, model_info in models.items():
#         gpu_count = model_info["gpu_count"]
#         gpu_carbon = gpu_count * T4_CARBON_PER_UNIT_KG

#         rows.append({
#             "Token Count": int(total_tokens),
#             "Model": model_name,
#             "GPU Count": gpu_count,
#             "GPU Carbon (kg)": gpu_carbon,
#             "Uncompressed TB Provisioned": uc_tb_used,
#             "Uncompressed Carbon (kg)": uc_carbon,
#             "Compressed TB Provisioned": c_tb_used,
#             "Compressed Carbon (kg)": c_carbon,
#             "Total Carbon (Uncompressed, kg)": uc_carbon + gpu_carbon,
#             "Total Carbon (Compressed, kg)": c_carbon + gpu_carbon,
#         })

rows = []

for total_tokens in token_counts:
    num_chunks = int(total_tokens // CHUNK_SIZE)

    # Raw text
    raw_text_bytes = total_tokens * TOKENS_PER_BYTE

    # Datastore (chunked storage)
    datastore_bytes = num_chunks * (CHUNK_SIZE + EMBEDDING_DIM * FLOAT32_SIZE + METADATA_BYTES_PER_CHUNK)

    # Total base data (both raw + chunks)
    total_datastore_bytes = raw_text_bytes + datastore_bytes

    # === FAISS Indexes ===

    # Flat index
    flat_index_bytes = EMBEDDING_DIM * FLOAT32_SIZE * num_chunks

    # IVF PQ128
    pq128_index_bytes = (128 * num_chunks) + (16384 * EMBEDDING_DIM * FLOAT32_SIZE)

    # IVF PQ256
    pq256_index_bytes = (256 * num_chunks) + (16384 * EMBEDDING_DIM * FLOAT32_SIZE)

    # IVF SQ8
    sq8_index_bytes = (EMBEDDING_DIM * num_chunks) + (16384 * EMBEDDING_DIM * FLOAT32_SIZE)

    # HNSW SQ8
    hnsw_index_bytes = (EMBEDDING_DIM + (128 * 2 * 4)) * num_chunks

    # === Total Sizes in TB ===

    def tb_total(index_bytes):
        return bytes_to_tb(total_datastore_bytes + index_bytes)

    total_tb_flat = tb_total(flat_index_bytes)
    total_tb_pq128 = tb_total(pq128_index_bytes)
    total_tb_pq256 = tb_total(pq256_index_bytes)
    total_tb_sq8 = tb_total(sq8_index_bytes)
    total_tb_hnsw = tb_total(hnsw_index_bytes)

    # === Carbon + SSD Calculations ===

    def storage_stats(tb):
        ssds, provisioned_tb, carbon = calculate_storage_impact(tb)
        return provisioned_tb, carbon

    flat_tb, flat_carbon = storage_stats(total_tb_flat)
    pq128_tb, pq128_carbon = storage_stats(total_tb_pq128)
    pq256_tb, pq256_carbon = storage_stats(total_tb_pq256)
    sq8_tb, sq8_carbon = storage_stats(total_tb_sq8)
    hnsw_tb, hnsw_carbon = storage_stats(total_tb_hnsw)

    # === Append results per model ===

    for model_name, model_info in models.items():
        gpu_count = model_info["gpu_count"]
        gpu_carbon = gpu_count * T4_CARBON_PER_UNIT_KG

        rows.append({
            "Token Count": int(total_tokens),
            "Model": model_name,
            "GPU Count": gpu_count,
            "GPU Carbon (kg)": gpu_carbon,

            "FAISS: Flat TB": flat_tb,
            "FAISS: PQ128 TB": pq128_tb,
            "FAISS: PQ256 TB": pq256_tb,
            "FAISS: SQ8 TB": sq8_tb,
            "FAISS: HNSW TB": hnsw_tb,

            "FAISS: Flat Carbon (kg)": flat_carbon,
            "FAISS: PQ128 Carbon (kg)": pq128_carbon,
            "FAISS: PQ256 Carbon (kg)": pq256_carbon,
            "FAISS: SQ8 Carbon (kg)": sq8_carbon,
            "FAISS: HNSW Carbon (kg)": hnsw_carbon,

            "Total Carbon (Flat)": flat_carbon + gpu_carbon,
            "Total Carbon (PQ128)": pq128_carbon + gpu_carbon,
            "Total Carbon (PQ256)": pq256_carbon + gpu_carbon,
            "Total Carbon (SQ8)": sq8_carbon + gpu_carbon,
            "Total Carbon (HNSW)": hnsw_carbon + gpu_carbon,
        })


# === Save to CSV ===

df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
df.to_csv("results/rag_model_gpu_total_impact.csv", index=False)

# # === Plot 1: Total Carbon (Uncompressed) ===

# plt.figure(figsize=(12, 6))
# for model in df["Model"].unique():
#     subset = df[df["Model"] == model]
#     plt.plot(subset["Token Count"], subset["Total Carbon (Uncompressed, kg)"],
#              marker='o', label=f"{model} (Uncompressed)")
# plt.title("Total Carbon Footprint (Uncompressed) by Model and Token Count")
# plt.xlabel("Token Count")
# plt.ylabel("Total Carbon (kg CO₂e)")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("results/total_carbon_uncompressed.png")
# plt.close()

# # === Plot 2: Total Carbon (Compressed) ===

# plt.figure(figsize=(12, 6))
# for model in df["Model"].unique():
#     subset = df[df["Model"] == model]
#     plt.plot(subset["Token Count"], subset["Total Carbon (Compressed, kg)"],
#              marker='o', label=f"{model} (Compressed)")
# plt.title("Total Carbon Footprint (Compressed) by Model and Token Count")
# plt.xlabel("Token Count")
# plt.ylabel("Total Carbon (kg CO₂e)")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("results/total_carbon_compressed.png")
# plt.close()

# === Plot 1: Total Carbon by FAISS Index Type (Grouped by Model) ===

plt.figure(figsize=(14, 8))
faiss_types = ["Flat", "PQ128", "PQ256", "SQ8", "HNSW"]

for model in df["Model"].unique():
    for faiss_type in faiss_types:
        subset = df[df["Model"] == model]
        plt.plot(
            subset["Token Count"],
            subset[f"Total Carbon ({faiss_type})"],
            marker='o',
            label=f"{model} ({faiss_type})"
        )

plt.title("Total Carbon Footprint by FAISS Index Type and Model")
plt.xlabel("Token Count")
plt.ylabel("Total Carbon (kg CO₂e)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/total_carbon_by_faiss_type_and_model.png")
plt.close()

# === Plot 2: Storage Provisioned (TB) by FAISS Index Type ===

plt.figure(figsize=(14, 8))
bar_width = 0.15
index = range(len(df["Token Count"].unique()))
token_ticks = sorted(df["Token Count"].unique())

for i, faiss_type in enumerate(faiss_types):
    tb_vals = []
    for token_count in token_ticks:
        avg = df[df["Token Count"] == token_count][f"FAISS: {faiss_type} TB"].mean()
        tb_vals.append(avg)
    positions = [x + i * bar_width for x in index]
    plt.bar(positions, tb_vals, width=bar_width, label=faiss_type)

plt.title("Average Provisioned Storage (TB) by FAISS Index Type")
plt.xlabel("Token Count")
plt.ylabel("Provisioned Storage (TB)")
plt.xticks([r + 2 * bar_width for r in index], [f"{int(tc/1e9)}B" for tc in token_ticks])
plt.legend()
plt.tight_layout()
plt.savefig("results/provisioned_storage_by_faiss_type.png")
plt.close()

# === Plot 3: Relative Carbon Reduction from Flat Index ===

plt.figure(figsize=(14, 8))
for model in df["Model"].unique():
    for faiss_type in ["PQ128", "PQ256", "SQ8", "HNSW"]:
        subset = df[df["Model"] == model]
        reduction = subset["Total Carbon (Flat)"] - subset[f"Total Carbon ({faiss_type})"]
        plt.plot(
            subset["Token Count"],
            reduction,
            marker='o',
            label=f"{model} vs Flat ({faiss_type})"
        )

plt.title("Carbon Reduction Compared to Flat FAISS Index")
plt.xlabel("Token Count")
plt.ylabel("Carbon Reduction (kg CO₂e)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/carbon_reduction_vs_flat.png")
plt.close()


# Set up the FAISS types and corresponding column names
faiss_types = ["Flat", "PQ128", "PQ256", "SQ8", "HNSW"]
carbon_columns = {
    "Flat": "FAISS: Flat Carbon (kg)",
    "PQ128": "FAISS: PQ128 Carbon (kg)",
    "PQ256": "FAISS: PQ256 Carbon (kg)",
    "SQ8": "FAISS: SQ8 Carbon (kg)",
    "HNSW": "FAISS: HNSW Carbon (kg)"
}

# Get unique token counts (sorted)
token_counts_sorted = sorted(df["Token Count"].unique())

# === Plot each FAISS index's carbon curve ===
plt.figure(figsize=(14, 8))

for faiss_type in faiss_types:
    y_vals = []
    for token_count in token_counts_sorted:
        avg = df[df["Token Count"] == token_count][carbon_columns[faiss_type]].mean()
        y_vals.append(avg)

    plt.plot(token_counts_sorted, y_vals, marker='o', label=faiss_type)

# Final plot formatting
plt.title("Carbon Emissions from Provisioned Storage by FAISS Index Type")
plt.xlabel("Token Count")
plt.ylabel("Storage Carbon (kg CO₂e)")
plt.grid(True)
plt.legend(title="FAISS Index Type")
plt.tight_layout()
plt.savefig("results/storage_carbon_by_faiss_type.png")
plt.close()


token_counts_sorted = sorted(df["Token Count"].unique())
x_labels = [f"{int(tc/1e9)}B" for tc in token_counts_sorted]
x = np.arange(len(token_counts_sorted))
bar_width = 0.15

# === Plotting ===
plt.figure(figsize=(14, 8))

for i, faiss_type in enumerate(faiss_types):
    values = []
    for token_count in token_counts_sorted:
        avg_carbon = df[df["Token Count"] == token_count][carbon_columns[faiss_type]].mean()
        values.append(avg_carbon)
    
    positions = x + i * bar_width
    plt.bar(positions, values, width=bar_width, label=faiss_type)

# === Formatting ===
plt.xlabel("Token Count")
plt.ylabel("Total Carbon (kg CO₂e)")
plt.title("Average Total Carbon Emissions by FAISS Index Type")
plt.xticks(x + bar_width * 2, x_labels)
plt.legend(title="FAISS Index Type")
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("results/total_carbon_grouped_bar.png")
plt.show()

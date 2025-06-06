# MASSIVEDS Carbon Footprint Calculator (T4 GPU Optimized)

import matplotlib.pyplot as plt

def calculate_carbon_footprint(
    num_inferences,
    inference_time_sec,  # Duration of one inference in seconds
    gpu_power_kw=0.07,   # T4 GPU average power draw in kilowatts (70W)
    pue=1.11,            # Power Usage Effectiveness (datacenter overhead factor)
    i_grid=0.234,        # Grid carbon intensity in kgCO2e/kWh

    ef_trans=0.003,        # Energy per GB of data transfer in kWh
    s_per_inference_gb=0.00001,  # Data retrieved per inference in GB (10 KB)
    i_route=0.234,         # Network route carbon intensity (assumed equal to i_grid)

    datastore_flops=2.478e20,     # Total FLOPs to build the datastore
    flops_per_kwh=5e10,           # Hardware efficiency in FLOPs per kWh
    datastore_total_inferences=1e9,  # Number of inferences using the datastore

    device_lifetime_years=4,  # Total expected device lifetime
    app_runtime_years=1,      # Duration the application is run
    die_area_mm2=400,         # Die area of SoC in mm^2
    ci_fab=0.5,               # Fab carbon intensity (kgCO2e/kWh)
    epa=0.1,                  # Energy per mm^2 manufactured (kWh/mm^2)
    gpa=0.05,                 # GHG per mm^2 (kgCO2e/mm^2)
    mpa=0.1,                  # Material carbon per mm^2 (kgCO2e/mm^2)
    yield_percent=0.9,        # Fabrication yield
    cps_dram=1.2,             # Carbon per GB of DRAM (kgCO2e/GB)
    capacity_dram_gb=16,      # DRAM capacity
    cps_ssd=0.5,              # Carbon per GB of SSD (kgCO2e/GB)
    capacity_ssd_gb=256,      # SSD capacity

    pretraining_emissions_tons=0.0  # Meta pretraining carbon (fixed)
):
    # Step 1: Execution Carbon per inference (using T4 GPU)
    e_gpu = gpu_power_kw * inference_time_sec / 3600  # kWh
    carbon_ex = i_grid * e_gpu * pue

    # Step 2: Transmission Carbon per inference
    e_trans = ef_trans * s_per_inference_gb
    carbon_tran = i_route * e_trans

    # Step 3: Datastore Construction Carbon (amortized)
    energy_index_kwh = datastore_flops / flops_per_kwh
    carbon_index_total = i_grid * energy_index_kwh
    carbon_index_per_inference = carbon_index_total / datastore_total_inferences

    # Step 4: Embodied Carbon (amortized)
    y = yield_percent
    e_soc = die_area_mm2 * (ci_fab * epa + gpa + mpa / y)
    e_dram = cps_dram * capacity_dram_gb
    e_ssd = cps_ssd * capacity_ssd_gb
    ecf_total = e_soc + e_dram + e_ssd
    ecf_amortized = (app_runtime_years / device_lifetime_years) * ecf_total
    carbon_embodied_per_inference = ecf_amortized / num_inferences

    # Step 5: Total
    total_per_inference = (
        carbon_ex + carbon_tran + carbon_index_per_inference + carbon_embodied_per_inference
    )
    # total_per_inference = (
    #     carbon_ex + carbon_tran + carbon_embodied_per_inference
    # )
    total_all_kg = total_per_inference * num_inferences
    total_all_tons = total_all_kg / 1000 + pretraining_emissions_tons

    return {
        "carbon_ex_per_inference": carbon_ex,
        "carbon_tran_per_inference": carbon_tran,
        "carbon_index_per_inference": carbon_index_per_inference,
        "carbon_embodied_per_inference": carbon_embodied_per_inference,
        "total_carbon_per_inference": total_per_inference,
        "total_carbon_all_inferences_kg": total_all_kg,
        "total_carbon_all_inferences_tons": total_all_tons
    }


# Example usage
if __name__ == "__main__":
    # https://huggingface.co/meta-llama/Meta-Llama-3-8B?utm_source=chatgpt.com
    # https://huggingface.co/meta-llama/Llama-2-13b?utm_source=chatgpt.com
    models = {
        "llama2_7b": {"inference_time": 2.0, "pretrain_tons": 31.22},
        "llama3_8b": {"inference_time": 2.3, "pretrain_tons": 390.0},
        "llama2_13b": {"inference_time": 3.5, "pretrain_tons": 62.44}
    }

    inference_counts = [14e9, 140e9, 350e9, 700e9, 1e12, 1.4e12]

    model_results = {}

    for model_name, values in models.items():
        tons_list = []
        print(f"\n=== {model_name.upper()} ===")
        for count in inference_counts:
            result = calculate_carbon_footprint(
                inference_time_sec=values["inference_time"],
                num_inferences=int(count),
                pretraining_emissions_tons=values["pretrain_tons"]
            )

            print(f"\nInferences: {int(count):,}")
            for k, v in result.items():
                print(f"{k}: {v}")

            tons_list.append(result["total_carbon_all_inferences_tons"])

        model_results[model_name] = tons_list

    # Plotting the results
    plt.figure(figsize=(12, 6))
    for model_name, tons in model_results.items():
        plt.plot(inference_counts, tons, marker='o', label=model_name)

    plt.title("Total Carbon Emissions vs Inference Count")
    plt.xlabel("Number of Inferences")
    plt.ylabel("Total Carbon Emissions (tons CO₂e)")
    plt.xticks(inference_counts, [f"{int(x/1e9)}B" for x in inference_counts])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # # Plotting the relative difference from LLAMA2_7B
    # baseline = model_results["llama2_7b"]
    # plt.figure(figsize=(12, 6))

    # for model_name, tons in model_results.items():
    #     if model_name == "llama2_7b":
    #         continue  # Skip baseline
    #     delta = [model - base for model, base in zip(tons, baseline)]
    #     plt.plot(inference_counts, delta, marker='o', label=f"{model_name} − llama2_7b")

    # plt.title("Carbon Emission Difference from LLAMA2_7B vs Inference Count")
    # plt.xlabel("Number of Inferences")
    # plt.ylabel("Emission Difference (tons CO₂e)")
    # plt.xticks(inference_counts, [f"{int(x/1e9)}B" for x in inference_counts])
    # plt.grid(True)
    # plt.axhline(0, color='gray', linestyle='--')  # Reference line
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

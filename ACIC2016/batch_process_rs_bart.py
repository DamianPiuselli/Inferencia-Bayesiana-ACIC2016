import pandas as pd
import os
import pymc as pm
import pymc_bart as pmb
import numpy as np
import arviz as az

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def save_metrics_to_csv(metrics, file_name="model_metrics.csv"):
    file_path = os.path.join(SCRIPT_DIR, "..", file_name)  # Save to project root
    
    # Check if the file exists
    file_exists = os.path.isfile(file_path)

    # Convert metrics dictionary to a DataFrame
    metrics_df = pd.DataFrame([metrics])

    # Append to the CSV file
    if file_exists:
        metrics_df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        metrics_df.to_csv(file_path, mode="w", header=True, index=False)


def process_covariables():
    x_path = os.path.join(SCRIPT_DIR, "acic_challenge_2016", "x.csv")
    X = pd.read_csv(x_path)

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X_encoded = X_encoded.astype("float32")

    return X_encoded


def pipe_model(y_df, X):

    z = y_df["z"].values
    y = y_df["y0"].where(
        z == 0, y_df["y1"]
    )  # respuesta observada depende del tratamiento

    y_std = y.std()  # Para estandarizar las metricas

    with pm.Model() as model1:
        # covariables + Z
        X_combined = pm.Data("X_combined", X_encoded.assign(treatment=z))

        # Define BART model
        mu = pmb.BART("mu", X_combined, Y=y, m=250)

        # Likelihood
        sigma = pm.InverseGamma("sigma", alpha=1, beta=2)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y, initval=y.std())

        # Sample
        trace = pm.sample()

        # sample posterior predictive
        X_treated = pm.set_data({"X_combined": X_encoded.assign(treatment=1.0)})
        mu_treated = pm.sample_posterior_predictive(trace, var_names=["mu"])

        X_control = pm.set_data({"X_combined": X_encoded.assign(treatment=0.0)})
        mu_control = pm.sample_posterior_predictive(trace, var_names=["mu"])

        # extract from trace
        pp_treated = az.extract(mu_treated.posterior_predictive)
        pp_control = az.extract(mu_control.posterior_predictive)

        ## SATE
        pp_sate = (pp_treated - pp_control).mean(dim="mu_dim_0")
        sate_values = pp_sate["mu"].values
        ground_truth_sate = (y_df["y1"] - y_df["y0"]).mean()

        sate_bias = (np.mean(sate_values) - ground_truth_sate) / y_std
        print(f"SATE Bias: {sate_bias}")

        sate_rmse = np.sqrt(np.mean((sate_values - ground_truth_sate) ** 2)) / y_std
        print(f"SATE RMSE: {sate_rmse}")

        sate_ci = az.hdi(sate_values, hdi_prob=0.95)
        sate_ci_length = (sate_ci[1] - sate_ci[0]) / y_std
        print(f"CI Length: {sate_ci_length}")

        sate_coverage = 1 if (sate_ci[0] < ground_truth_sate < sate_ci[1]) else 0
        print(f"SATE Coverage: {sate_coverage}")

        ## ATT
        treated_indices = np.where(z == 1)[0]
        pp_att = (
            pp_treated["mu"].values[treated_indices, :]
            - pp_control["mu"].values[treated_indices, :]
        )

        att_values = pp_att.mean(axis=0)

        ground_truth_att = (
            y_df.loc[treated_indices, "y1"] - y_df.loc[treated_indices, "y0"]
        ).mean()

        att_bias = (np.mean(pp_att) - ground_truth_att) / y_std
        print(f"ATT Bias: {att_bias}")

        att_rmse = np.sqrt(np.mean((pp_att - ground_truth_att) ** 2)) / y_std
        print(f"ATT RMSE: {att_rmse}")

        att_ci = az.hdi(att_values, hdi_prob=0.95)
        att_ci_length = (att_ci[1] - att_ci[0]) / y_std
        print(f"CI Length: {att_ci_length}")

        att_coverage = 1 if (att_ci[0] < ground_truth_att < att_ci[1]) else 0
        print(f"ATT Coverage: {att_coverage}")

        ## PEHE
        ite_estimates = pp_treated["mu"].values - pp_control["mu"].values
        ground_truth_ite = y_df["y1"] - y_df["y0"]

        pehe = (
            np.sqrt(np.mean((ite_estimates - ground_truth_ite.values[:, None]) ** 2))
            / y_std
        )
        print(f"PEHE: {pehe}")

        metrics = {
            "SATE_Bias": sate_bias,
            "SATE_RMSE": sate_rmse,
            "SATE_CI_Length": sate_ci_length,
            "SATE_Coverage": sate_coverage,
            "ATT_Bias": att_bias,
            "ATT_RMSE": att_rmse,
            "ATT_CI_Length": att_ci_length,
            "ATT_Coverage": att_coverage,
            "PEHE": pehe,
        }

        save_metrics_to_csv(metrics)


def process_csv_files(folder_path, skip_first_n=0):
    processed_count = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            processed_count += 1
            if processed_count <= skip_first_n:
                print(f"Skipping file {file_name} ({processed_count}/{skip_first_n})")
                continue

            file_path = os.path.join(folder_path, file_name)

            df = pd.read_csv(file_path)
            expected_columns = ["z", "y0", "y1", "mu0", "mu1"]

            if list(df.columns) == expected_columns:
                print(f"Successfully loaded {file_name}, {df.columns}")
                pipe_model(df, X_encoded)
            else:
                print(f"Warning: {file_name} does not have the expected header")


if __name__ == "__main__":
    # Define the folder containing the .csv files
    y_folder_path = os.path.join(SCRIPT_DIR, "sample_data_from_cf_all")

    X_encoded = process_covariables()

    process_csv_files(y_folder_path, skip_first_n=44)

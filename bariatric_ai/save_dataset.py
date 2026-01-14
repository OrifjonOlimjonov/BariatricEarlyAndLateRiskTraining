import os
from pathlib import Path

# Eslatma:
# Sizning faylingiz loyihada qaysi nom bilan turgan bo‘lsa, import ham shunga mos bo‘lishi kerak.
# Agar funksiyangiz data_generate.py ichida bo‘lsa:
from data_generate import generate_synthetic_bariatric
# Agar sizda haqiqiy fayl nomi generate_synthetic_data.py bo‘lsa, yuqoridagini komment qiling va buni oching:
# from generate_synthetic_data import generate_synthetic_bariatric


def main():
    n_patients = 10_000
    seed = 42

    out_dir = Path("data_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df, outcomes = generate_synthetic_bariatric(n_patients=n_patients, seed=seed)

    long_path = out_dir / f"bariatric_long_{n_patients}_seed{seed}.csv"
    out_path = out_dir / f"bariatric_outcomes_{n_patients}_seed{seed}.csv"

    long_df.to_csv(long_path, index=False)
    outcomes.to_csv(out_path, index=False)

    # Kichik README (tezis/reproducibility uchun)
    readme_path = out_dir / f"README_bariatric_{n_patients}_seed{seed}.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("Synthetic bariatric dataset\n")
        f.write(f"n_patients={n_patients}\nseed={seed}\n\n")
        f.write("Files:\n")
        f.write(f"- {long_path.name}: longitudinal measurements (patient_id x timepoint)\n")
        f.write(f"- {out_path.name}: patient-level outcomes/labels\n")

    print("Saved:")
    print(" -", long_path)
    print(" -", out_path)
    print(" -", readme_path)


if __name__ == "__main__":
    main()
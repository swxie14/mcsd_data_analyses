from decoding.test_decoder import test_decoder, compare_decoders
import pandas as pd
import numpy as np

from decoding.option_ccgp_mcsd import option_ccgp_mcsd



def test_decoders(decoder : str, n_perms : int=1000):
    # Example 1: Test single region
    #test_results = test_decoder(decoder, n_perms=n_perms)
    
    # Example 2: Compare across all regions
    print(f"\n\nExample 2: Comparing all regions for {decoder} decoder")
    results = compare_decoders(decoder, n_perms=n_perms)



def option_ccgp():
    regions = ["ACC", "A", "H", "OF", "PT", "CM", "SMA", "PRV", "IFG"]
    context_combos = [
        (1, 2, 3),  # train 1 & 2, test 3
        (1, 3, 2),  # train 1 & 3, test 2
        (2, 3, 1),  # train 2 & 3, test 1
    ]

    summary_records = []

    for option_1, option_2, test_option in context_combos:
        ctx_label = f"{option_1}&{option_2}->{test_option}"
        print(f"\nRunning context {ctx_label}")
        for region in regions:
            accs, within_acc = option_ccgp_mcsd(region=region,
                                                option_1=option_1,
                                                option_2=option_2,
                                                test_option=test_option)
            summary_records.append({
                "Region": region,
                "Context": ctx_label,
                "Accuracy": np.mean(accs),
                "WithinClassAcc": np.mean(within_acc)
            })

    # build summary table
    summary_df = pd.DataFrame(summary_records)
    accuracy_table = summary_df.pivot(index="Region", columns="Context", values="Accuracy")

    print("\n==============================")
    print("Accuracy by Region and Context")
    print("==============================")
    print(accuracy_table.to_string(float_format="{:.3f}".format))


if __name__ == "__main__":
   #test_decoders('second_choice')
   option_ccgp()

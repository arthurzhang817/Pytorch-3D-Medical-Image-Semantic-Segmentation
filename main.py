import argparse

from train import train

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Specify: mode and configuration file path.")

    parser.add_argument(
        "-m",
        "--mode",
        nargs="?",
        choices=['train', 'val', 'test', 'demo'],
        type=str,
        default="train",
        help="Specify the mode here: trian/val/test/demo"
    )

    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        type=str,
        default="config/prostateCT_deeper3dresunet_train.yml",
        help="Specify the path of configuration file here."
    )

    args = parser.parse_args()

    locals()[args.mode](args.config)
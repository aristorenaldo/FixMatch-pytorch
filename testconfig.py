from config_utils import config_parser
import argparse

parser = argparse.ArgumentParser(description='Semi G-SSL Training')
parser.add_argument('--path', '-p', type=str, help='Config Path')
cli_parser = parser.parse_args()
print(cli_parser.path)

config = config_parser('./config/semi_gssl_cifar10_4000_default.yaml', cli_parser.path)
args = config.get()

# set save_name
args.save_name += f'_{args.arch}_{args.dataset}_{args.num_labels}'
print(args.__dict__)
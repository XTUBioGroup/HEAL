import argparse
import os

from train import Trainer


def main():
    parser = argparse.ArgumentParser(description="Run HEAL basic training and validation.")

    # Paths
    parser.add_argument('--data_root', type=str, default='Dataset/Final_Datas',
                        help='Directory containing graph file and dataset_split folder.')
    parser.add_argument('--runs_dir', type=str, default='Experiment/runs/publicversion',
                        help='Directory to save training runs.')
    parser.add_argument('--graph_filename', type=str, default='hetero_graph_v8.0.pt',
                        help='Graph filename under data_root.')
    parser.add_argument('--train_path', type=str, default=None,
                        help='Optional custom training CSV path.')
    parser.add_argument('--test_path', type=str, default=None,
                        help='Optional custom testing CSV path.')

    # Device
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device ID, e.g. "0".')

    # Model and training
    parser.add_argument('--h_dim', type=int, default=128)
    parser.add_argument('--num_attention_heads', type=int, default=4)
    parser.add_argument('--num_hgt_layers', type=int, default=2)
    parser.add_argument('--num_neighbors', type=int, nargs='+', default=[128, 64])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--loss_beta', type=float, default=0.1)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    print(f"Set CUDA_VISIBLE_DEVICES = {args.device}")

    trainer = Trainer(args)
    trainer.run()


if __name__ == '__main__':
    main()

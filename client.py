import argparse
import os
from multiprocessing.connection import Client, Connection

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="140.112.30.56")
    parser.add_argument("--port", type=int, default=9092)
    return parser.parse_args()


def monitor_progress(conn: Connection):
    with tqdm(total=conn.recv()) as pbar:
        while (n := conn.recv()) > 0:
            pbar.update(n)


def create(dataset_name: str, images: list[str], conn: Connection):
    conn.send(["create", dataset_name])

    if not conn.recv():
        print(f"Oops! Dataset '{dataset_name}' already exists.")
        return

    conn.send(len(images))
    for image in images:
        extension = image.split(".")[-1]
        with open(image, mode="rb") as file:
            contents = file.read()
        conn.send(extension)
        conn.send(contents)

    print("store images...")
    monitor_progress(conn)
    print("downsample images...")
    monitor_progress(conn)


def generate_masks(dataset_name: str, conn: Connection):
    conn.send(["generate-masks", dataset_name])

    print("generate masks...")
    monitor_progress(conn)


def loo(dataset_name: str, conn: Connection):
    conn.send(["loo", dataset_name])

    print("leave-one-out stage 1...")
    monitor_progress(conn)
    print("leave-one-out stage 2...")
    monitor_progress(conn)


def ls(conn: Connection):
    conn.send(["ls"])

    for dataset_name in conn.recv():
        print(dataset_name)


def pred_poses(dataset_name: str, conn: Connection):
    conn.send(["pred-poses", dataset_name])

    print("inference...")
    monitor_progress(conn)
    print("compute global alignment...")
    monitor_progress(conn)


def render(dataset_name: str, output_dir: str, conn: Connection):
    conn.send(["render", dataset_name])

    print("render video frames...")
    monitor_progress(conn)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "renders.mp4"), mode="wb") as file:
        file.write(conn.recv())


def train_gs(dataset_name: str, conn: Connection):
    conn.send(["train-gs", dataset_name])

    print("train coarse 3DGS...")
    monitor_progress(conn)


def train_lora(dataset_name: str, conn: Connection):
    conn.send(["train-lora", dataset_name])

    print("fine-tune Gaussian repair model with LoRA...")
    monitor_progress(conn)


def train_repair(dataset_name: str, conn: Connection):
    conn.send(["train-repair", dataset_name])

    print("repair the 3DGS representation...")
    monitor_progress(conn)


def main():
    args = parse_args()
    with Client(address=(args.ip, args.port)) as conn:
        while True:
            print("$ ", end="")
            argv = input().split()
            command_name = argv[0]
            if command_name == "create":
                create(dataset_name=argv[1], images=argv[2:], conn=conn)
            elif command_name == "generate-masks":
                generate_masks(dataset_name=argv[1], conn=conn)
            elif command_name == "loo":
                loo(dataset_name=argv[1], conn=conn)
            elif command_name == "ls":
                ls(conn=conn)
            elif command_name == "pred-poses":
                pred_poses(dataset_name=argv[1], conn=conn)
            elif command_name == "render":
                render(dataset_name=argv[1], output_dir=argv[2], conn=conn)
            elif command_name == "train-gs":
                train_gs(dataset_name=argv[1], conn=conn)
            elif command_name == "train-lora":
                train_lora(dataset_name=argv[1], conn=conn)
            elif command_name == "train-repair":
                train_repair(dataset_name=argv[1], conn=conn)


if __name__ == "__main__":
    main()

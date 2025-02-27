import argparse
import os
from multiprocessing.connection import Connection, Listener

from PIL import Image

import generate_masks as gm
import leave_one_out_stage1 as loo1
import leave_one_out_stage2 as loo2
import pred_poses as pp
import render as rd
import train_gs as tg
import train_lora as tl
import train_repair as tr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="140.112.30.56")
    parser.add_argument("--port", type=int, default=9092)
    return parser.parse_args()


def create(dataset_name: str, conn: Connection):
    dataset_dir = os.path.join("data", dataset_name)
    images_dir = os.path.join(dataset_dir, "images")

    try:
        os.makedirs(images_dir)
        conn.send(True)
    except FileExistsError:
        conn.send(False)
        return

    num_images = conn.recv()
    conn.send(num_images)
    for i in range(num_images):
        extension, contents = conn.recv(), conn.recv()
        with open(os.path.join(images_dir, f"{i:03}.{extension}"), mode="wb") as file:
            file.write(contents)
        conn.send(1)
    conn.send(0)

    for filename in (f"sparse_{num_images}.txt", "sparse_test.txt"):
        with open(os.path.join(dataset_dir, filename), mode="w") as file:
            file.writelines(f"{i}\n" for i in range(num_images))

    factors = (2, 4, 8)
    conn.send(len(factors) * num_images)
    for factor in factors:
        resized_images_dir = f"{images_dir}_{factor}"
        os.mkdir(resized_images_dir)
        for image_name in sorted(os.listdir(images_dir)):
            image = Image.open(os.path.join(images_dir, image_name))
            orig_w, orig_h = image.size[0], image.size[1]
            resolution = round(orig_w / factor), round(orig_h / factor)
            image = image.resize(resolution)
            image.save(os.path.join(resized_images_dir, image_name))
            conn.send(1)
    conn.send(0)


def generate_masks(dataset_name: str, conn: Connection):
    dataset_dir = os.path.join("data", dataset_name)
    sparse_num = len(os.listdir(os.path.join(dataset_dir, "images")))
    argv = ["-s", dataset_dir, "--sparse_num", str(sparse_num)]
    gm.main(argv, conn)


def loo(dataset_name: str, conn: Connection):
    dataset_dir = os.path.join("data", dataset_name)
    sparse_num = len(os.listdir(os.path.join(dataset_dir, "images")))
    argv = [
        "-s", dataset_dir,
        "-m", f"output/gs_init/{dataset_name}_loo",
        "-r", "8", "--sparse_view_num", str(sparse_num), "--sh_degree", "2",
        "--init_pcd_name", "dust3r_4",
        "--dust3r_json", f"output/gs_init/{dataset_name}/refined_cams.json",
        "--white_background", "--random_background", "--use_dust3r",
    ]
    loo1.main(argv, conn)
    loo2.main(argv, conn)


def ls(conn: Connection):
    conn.send(os.listdir("data"))


def pred_poses(dataset_name: str, conn: Connection):
    dataset_dir = os.path.join("data", dataset_name)
    sparse_num = len(os.listdir(os.path.join(dataset_dir, "images")))
    argv = ["-s", dataset_dir, "--sparse_num", str(sparse_num)]
    pp.main(argv, conn)


def render(dataset_name: str, conn: Connection):
    dataset_dir = os.path.join("data", dataset_name)
    sparse_num = len(os.listdir(os.path.join(dataset_dir, "images")))
    argv = [
        "-m", f"output/gs_init/{dataset_name}",
        "--sparse_view_num", str(sparse_num), "--sh_degree", "2",
        "--init_pcd_name", "dust3r_4",
        "--white_background", "--render_path", "--use_dust3r",
        "--load_ply", f"output/gaussian_object/{dataset_name}/save/last.ply",
    ]
    rd.main(argv, conn)


def train_gs(dataset_name: str, conn: Connection):
    dataset_dir = os.path.join("data", dataset_name)
    sparse_num = len(os.listdir(os.path.join(dataset_dir, "images")))
    argv = [
        "-s", dataset_dir,
        "-m", f"output/gs_init/{dataset_name}",
        "-r", "8", "--sparse_view_num", str(sparse_num), "--sh_degree", "2",
        "--init_pcd_name", "dust3r_4",
        "--white_background", "--random_background", "--use_dust3r",
    ]
    tg.main(argv, conn)


def train_lora(dataset_name: str, conn: Connection):
    dataset_dir = os.path.join("data", dataset_name)
    sparse_num = len(os.listdir(os.path.join(dataset_dir, "images")))
    argv = [
        "--exp_name", f"controlnet_finetune/{dataset_name}",
        "--prompt", "xxy5syt00", "--sh_degree", "2", "--resolution", "8", "--sparse_num", str(sparse_num),
        "--data_dir", dataset_dir,
        "--gs_dir", f"output/gs_init/{dataset_name}",
        "--loo_dir", f"output/gs_init/{dataset_name}_loo",
        "--bg_white", "--sd_locked", "--train_lora", "--use_prompt_list",
        "--add_diffusion_lora", "--add_control_lora", "--add_clip_lora", "--use_dust3r",
    ]
    tl.main(argv, conn)


def train_repair(dataset_name: str, conn: Connection):
    dataset_dir = os.path.join("data", dataset_name)
    sparse_num = len(os.listdir(os.path.join(dataset_dir, "images")))
    argv = [
        "--config", "configs/gaussian-object-colmap-free.yaml",
        "--train", "--gpu", "0",
        f"tag=\"{dataset_name}\"",
        f"system.init_dreamer=\"output/gs_init/{dataset_name}\"",
        f"system.exp_name=\"output/controlnet_finetune/{dataset_name}\"",
        "system.refresh_size=8",
        f"data.data_dir=\"data/{dataset_name}\"",
        "data.resolution=8",
        f"data.sparse_num={sparse_num}",
        "data.prompt=\"a photo of a xxy5syt00\"",
        f"data.json_path=\"output/gs_init/{dataset_name}/refined_cams.json\"",
        "data.refresh_size=8",
        "system.sh_degree=2",
    ]
    tr.main(argv, conn)


def serve_conn(conn: Connection):
    while True:
        argv: list[str] = conn.recv()
        command_name = argv[0]
        if command_name == "create":
            create(dataset_name=argv[1], conn=conn)
        elif command_name == "generate-masks":
            generate_masks(dataset_name=argv[1], conn=conn)
        elif command_name == "loo":
            loo(dataset_name=argv[1], conn=conn)
        elif command_name == "ls":
            ls(conn=conn)
        elif command_name == "pred-poses":
            pred_poses(dataset_name=argv[1], conn=conn)
        elif command_name == "render":
            render(dataset_name=argv[1], conn=conn)
        elif command_name == "train-gs":
            train_gs(dataset_name=argv[1], conn=conn)
        elif command_name == "train-lora":
            train_lora(dataset_name=argv[1], conn=conn)
        elif command_name == "train-repair":
            train_repair(dataset_name=argv[1], conn=conn)


def main():
    args = parse_args()
    with Listener(address=(args.ip, args.port)) as listener:
        with listener.accept() as conn:
            serve_conn(conn)


if __name__ == "__main__":
    main()

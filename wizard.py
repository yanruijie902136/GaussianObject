import os
import shutil
import tkinter as tk
from tkinter import filedialog, ttk

from generate_masks import GenerateMasks
from leave_one_out_stage1 import LeaveOneOutStage1
from leave_one_out_stage2 import LeaveOneOutStage2
from pred_poses import PredPoses
from preprocess.downsample import Downsample
from render import Render
from train_gs import TrainGS
from train_lora import TrainLoRA
from train_repair import TrainRepair

from progress_bar import *

PROGRESS_BAR_LENGTH = 200


class Wizard(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("GaussianObject Wizard")
        self.geometry("1000x800")

        container = tk.Frame(master=self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.data = {}

        self.steps = {}
        for Step in (MainMenuStep, PrepareDatasetStep, PreviewMasksStep, TrainStep, RenderStep):
            step = Step(master=container, wizard=self)
            step.grid(row=0, column=0, sticky="nsew")
            self.steps[Step] = step

        self.show_step(MainMenuStep)

    def show_step(self, Step):
        step = self.steps[Step]
        step.tkraise()
        step.on_tkraise()


class WizardStep(tk.Frame):
    def __init__(self, master, wizard):
        super().__init__(master)
        self.wizard = wizard
        self.create_widgets()

    def create_widgets(self):
        raise NotImplementedError

    def on_tkraise(self):
        pass


class MainMenuStep(WizardStep):
    def create_widgets(self):
        lbl_title = tk.Label(master=self, text="GaussianObject")
        lbl_title.pack()

        lbl_subtitle = tk.Label(
            master=self, text="High-Quality 3D Object Reconstruction from Four Views with Gaussian Splatting"
        )
        lbl_subtitle.pack()

        btn_start = tk.Button(
            master=self, text="start", command=lambda: self.wizard.show_step(PrepareDatasetStep)
        )
        btn_start.pack()

        btn_quit = tk.Button(master=self, text="quit", command=self.wizard.quit)
        btn_quit.pack()


class PrepareDatasetStep(WizardStep):
    def create_widgets(self):
        lbl_dataset_name = tk.Label(master=self, text="dataset name:")
        lbl_dataset_name.pack()
        self.ent_dataset_name = tk.Entry(master=self)
        self.ent_dataset_name.pack()
        self.ent_dataset_name.bind("<Return>", self.on_return)

        lbl_create_folder = tk.Label(master=self, text="Create dataset folder...")
        lbl_create_folder.pack()
        self.pbr_create_folder = ttk.Progressbar(master=self, length=PROGRESS_BAR_LENGTH)
        self.pbr_create_folder.pack()

        lbl_downsample = tk.Label(master=self, text="Downsample images...")
        lbl_downsample.pack()
        self.pbr_downsample = ttk.Progressbar(master=self, length=PROGRESS_BAR_LENGTH)
        self.pbr_downsample.pack()

        lbl_generate_masks = tk.Label(master=self, text="Generate masks...")
        lbl_generate_masks.pack()
        self.pbr_generate_masks = ttk.Progressbar(master=self, length=PROGRESS_BAR_LENGTH)
        self.pbr_generate_masks.pack()

        self.btn_next = tk.Button(master=self, text="next", command=lambda: self.wizard.show_step(PreviewMasksStep))
        self.btn_next.pack()
        self.btn_next.config(state="disabled")

    def on_tkraise(self):
        self.btn_next.config(state="disabled")

    def on_return(self, event):
        dataset_name = self.ent_dataset_name.get()
        if not dataset_name:
            return
        source_path = os.path.join("data", dataset_name)

        image_paths = filedialog.askopenfilenames(
            filetypes=(
                ("png", "*.png"), ("jpeg", "*.jpeg"), ("jpg", "*.jpg")
            )
        )
        if not image_paths:
            return
        sparse_num = len(image_paths)

        self.wizard.data["source_path"] = source_path
        self.wizard.data["sparse_num"] = sparse_num

        self.create_folder(source_path, image_paths)
        Downsample(self.wizard, self.pbr_downsample).main(source_path)
        GenerateMasks(self.wizard, self.pbr_generate_masks).main(source_path, sparse_num)

        self.btn_next.config(state="normal")

    def create_folder(self, source_path, image_paths):
        shutil.rmtree(source_path, ignore_errors=True)
        images_dir = os.path.join(source_path, "images")
        os.makedirs(images_dir)

        sparse_num = len(image_paths)

        set_progress_bar(self.wizard, self.pbr_create_folder, maximum=sparse_num)

        for i, image_path in enumerate(image_paths):
            new_image_path = os.path.join(images_dir, f"{i:04}.png")
            shutil.copyfile(image_path, new_image_path)

            progress_bar_step()

        clear_progress_bar()

        for filename in (f"sparse_{sparse_num}.txt", "sparse_test.txt"):
            with open(os.path.join(source_path, filename), mode="w") as file:
                file.writelines(f"{i}\n" for i in range(sparse_num))


class PreviewMasksStep(WizardStep):
    def create_widgets(self):
        btn_ok = tk.Button(master=self, text="ok", command=lambda: self.wizard.show_step(TrainStep))
        btn_ok.pack()

        btn_discard = tk.Button(master=self, text="discard", command=self.discard)
        btn_discard.pack()

    def discard(self):
        shutil.rmtree(self.wizard.data["source_path"])
        self.wizard.show_step(PrepareDatasetStep)


class TrainStep(WizardStep):
    def create_widgets(self):
        lbl_title = tk.Label(master=self, text="Train the model")
        lbl_title.pack()

        self.btn_train = tk.Button(master=self, text="train", command=self.train)
        self.btn_train.pack()

        lbl_pred_poses = tk.Label(master=self, text="Generate coarse poses...")
        lbl_pred_poses.pack()
        self.pbr_pred_poses = ttk.Progressbar(master=self, length=PROGRESS_BAR_LENGTH)
        self.pbr_pred_poses.pack()

        lbl_train_gs = tk.Label(master=self, text="Train coarse 3DGS...")
        lbl_train_gs.pack()
        self.pbr_train_gs = ttk.Progressbar(master=self, length=PROGRESS_BAR_LENGTH)
        self.pbr_train_gs.pack()

        lbl_loo_stage_1 = tk.Label(master=self, text="Leave one out stage 1...")
        lbl_loo_stage_1.pack()
        self.pbr_loo_stage_1 = ttk.Progressbar(master=self, length=PROGRESS_BAR_LENGTH)
        self.pbr_loo_stage_1.pack()

        lbl_loo_stage_2 = tk.Label(master=self, text="Leave one out stage 2...")
        lbl_loo_stage_2.pack()
        self.pbr_loo_stage_2 = ttk.Progressbar(master=self, length=PROGRESS_BAR_LENGTH)
        self.pbr_loo_stage_2.pack()

        lbl_train_lora = tk.Label(master=self, text="LoRA fune-tuning...")
        lbl_train_lora.pack()
        self.pbr_train_lora = ttk.Progressbar(master=self, length=PROGRESS_BAR_LENGTH)
        self.pbr_train_lora.pack()

        lbl_train_repair = tk.Label(master=self, text="Train Gaussian repair model...")
        lbl_train_repair.pack()
        self.pbr_train_repair = ttk.Progressbar(master=self, length=PROGRESS_BAR_LENGTH)
        self.pbr_train_repair.pack()

        self.btn_render = tk.Button(master=self, text="render", command=lambda: self.wizard.show_step(RenderStep))
        self.btn_render.pack()
        self.btn_render.config(state="disabled")

    def train(self):
        self.btn_train.config(state="disabled")

        scripts = [
            PredPoses(wizard=self, progress_bar=self.pbr_pred_poses),
            TrainGS(wizard=self, progress_bar=self.pbr_train_gs),
            LeaveOneOutStage1(wizard=self, progress_bar=self.pbr_loo_stage_1),
            LeaveOneOutStage2(wizard=self, progress_bar=self.pbr_loo_stage_2),
            TrainLoRA(wizard=self, progress_bar=self.pbr_train_lora),
            TrainRepair(wizard=self, progress_bar=self.pbr_train_repair),
        ]

        for script in scripts:
            script.main(self.wizard.data["source_path"], self.wizard.data["sparse_num"])

        self.btn_render.config(state="normal")


class RenderStep(WizardStep):
    def create_widgets(self):
        self.btn_render = tk.Button(master=self, text="render", command=self.render)
        self.btn_render.pack()

        self.pbr_render = ttk.Progressbar(master=self, length=PROGRESS_BAR_LENGTH)
        self.pbr_render.pack()

        self.btn_finish = tk.Button(master=self, text="finish", command=self.wizard.quit)
        self.btn_finish.pack()
        self.btn_finish.config(state="disabled")

    def render(self):
        Render(wizard=self, progress_bar=self.pbr_render).main(
            self.wizard.data["source_path"], self.wizard.data["sparse_num"]
        )
        self.btn_finish.config(state="normal")


if __name__ == "__main__":
    wizard = Wizard()
    wizard.mainloop()

"""Exactly-one-line coordinate-only T4 autopsy. No GPU call without explicit flags."""
import modal
from modal_inkvae import image,volume,repo,require_opt_in

app=modal.App('diffink-english-one-line-autopsy')
image=image.add_local_file(str(repo/'configs/vae_iam_autopsy_geometry.yaml'),'/app/configs/vae_iam_autopsy_geometry.yaml')

@app.function(image=image,volumes={'/data':volume},gpu='T4',cpu=4,memory=16384,timeout=900,retries=0,max_containers=1)
def run_geometry():
    from iam_tools.autopsy import train_geometry
    try:
        return train_geometry('/app/configs/vae_iam_autopsy_geometry.yaml','/app',allow_experimental=True)
    finally:volume.commit()

@app.local_entrypoint()
def main(train:bool=False,allow_experimental:bool=False):
    if not require_opt_in(train,allow_experimental):
        print('Not launched. Geometry only: one T4, one line, at most 1000 steps / 600 loop seconds.');return
    print(run_geometry.remote())

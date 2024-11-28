import torch
from diffusers import DiffusionPipeline
import os
from datetime import datetime
import argparse

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--exp", type=str, default=None)
	parser.add_argument("--inf_steps", type=int, default=100)
	parser.add_argument("--seeds", type=int, default=42)
	parser.add_argument("--scale", type=float, default=6.0)
	parser.add_argument("--samples", type=int, default=20)
	parser.add_argument("--prompt", type=str, default="a photo of <s2*> pine in the desert")
	parser.add_argument(
        "--modifier_token",
        type=str,
        default="<s1*>+<s2*>+<c1*>+<c2*>+<c3*>+<c4*>",
        help="A token to use as a modifier for the concept.",
    )

	args = parser.parse_args()
	
	prompts = [
		args.prompt
		]

	model_path = f'models/{args.exp}/'
	LOG_DIR=f''

	model_id = f"{model_path+LOG_DIR}"

	out_path = f'Results/{args.exp}_{args.inf_steps}_{args.seeds}_{args.scale}_{args.samples}/'

	isExist = os.path.exists(out_path)
	if not isExist:
  		os.makedirs(out_path)
		
	osExist = os.path.exists(f'{out_path}/{LOG_DIR}')

	if not osExist:
  		os.makedirs(f'{out_path}/{LOG_DIR}')
	
	pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", low_cpu_mem_usage=False, torch_dtype=torch.float16, safety_checker=None).to("cuda")
	pipe.unet.load_attn_procs(f"{model_id}", weight_name="pytorch_custom_diffusion_weights.bin")
	for modifier_token in args.modifier_token.split("+"):
		pipe.load_textual_inversion(f"{model_id}", weight_name=f"{modifier_token}.bin")

	for prompt in prompts:
		n_samples = args.samples
		path = f'{out_path}'

		for ind in range(n_samples):
			out = pipe(prompt)
			out.images[0].save(f"{path}/{prompt}_{ind}.png")

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import argparse
import numpy as np
def plot_rewards(args):
	plt.ioff()
	fig = plt.figure()
	rewards = np.load(args.model_path+"/testing_rewards.npy")
	epochs = 5 * np.arange(rewards.size)
	plt.xlabel('epochs')
	plt.ylabel('mean rewards')
	plt.plot(epochs, rewards)
	plt.show()
	fig.savefig("reward_curve.png", dpi=fig.dpi)

def main():
	parser = argparse.ArgumentParser(description="Hyperparameters for UpsideDown RL")
	parser.add_argument("--model_path", type=str)
	args = parser.parse_args()
	plot_rewards(args)


if __name__ == "__main__":
	main()
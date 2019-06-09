import click
from datetime import datetime


@click.command()
@click.argument("input-path", type=str)
@click.option("-s", "--output-path", type=str,
              default=datetime.now().strftime("../data/fixed-submission-%d-%m-%y-%H-%M-%S.txt"))
def main(input_path, output_path):
    with open(input_path, 'r') as input_file, open(output_path, 'w') as output_file:
        for line in input_file:
            epsilon, C, gamma, samples = line.split(";")
            fixed_samples = ",".join([str(int(sample) + 1) for sample in samples.split(",")])
            output_file.write(f"{epsilon};{C};{gamma};{fixed_samples}\n")
    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    main()

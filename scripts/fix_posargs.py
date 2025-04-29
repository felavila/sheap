import sys

def main():
    args = sys.argv[1:]
    new_args = []
    i = 0
    while i < len(args):
        if args[i] == "-E":
            # Replace -E with --extras
            new_args.append("--extras")
            if i + 1 < len(args):
                new_args.append(args[i + 1])
                i += 1
        else:
            new_args.append(args[i])
        i += 1
    # Output the corrected arguments
    print(" ".join(new_args))

if __name__ == "__main__":
    main()
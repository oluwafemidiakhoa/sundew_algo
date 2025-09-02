import argparse, json
from .demo import run_demo

def main():
    p = argparse.ArgumentParser(description="Sundew Algorithm CLI")
    p.add_argument("--demo", action="store_true", help="Run the interactive demo")
    p.add_argument("--events", type=int, default=40, help="Number of demo events")
    p.add_argument("--temperature", type=float, default=0.1, help="Gating temperature (0=hard)")
    p.add_argument("--save", type=str, default="sundew_results.json", help="Save demo results to JSON")
    args = p.parse_args()

    if args.demo:
        out = run_demo(args.events, args.temperature)
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n💾 Results saved to {args.save}")
    else:
        p.print_help()

if __name__ == "__main__":
    main()

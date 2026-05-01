MODE=${1:-finetuned}

if [ "$MODE" = "both" ]; then
	echo "Running evaluation for base then fine-tuned model..."
fi

python scripts/evaluate.py --mode "$MODE"

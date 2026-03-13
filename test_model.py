import argparse
import os
import pickle
import pandas as pd

from train_model import FEATURES


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Вводишь 10 признаков -> модель предсказывает density (выход: q)."
	)
	parser.add_argument("--model", default="train_model.pkl")
	args = parser.parse_args()

	if not os.path.exists(args.model):
		print(
			f"Файл модели не найден: {args.model}. "
			"Сначала запусти train_model.py, чтобы создать train_model.pkl"
		)
		return 2

	with open(args.model, "rb") as f:
		model = pickle.load(f)

	print("\nВвод признаков (как в train_model.py). Можно вводить числа с запятой, например 1,23 или с точкой, например 1.23. Для выхода введи q.\n")
	row = {}
	for feature in FEATURES:
		while True:
			s = input(f"{feature}: ").strip()
			if s.lower() in {"q", "quit", "exit"}:
				print("\nВыход.")
				return 130
			try:
				row[feature] = float(s.replace(",", "."))
				break
			except ValueError:
				print("Некорректное число. Попробуй снова.")

	X = pd.DataFrame([row], columns=FEATURES)
	pred = model.predict(X)[0]
	print(f"\nПрогноз density: {float(pred):.6f}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())


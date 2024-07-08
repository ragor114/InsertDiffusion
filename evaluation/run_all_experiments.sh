#!/bin/bash

source ~/venv/bin/activate

# rm -r results

python evaluate.py --exp_name "bg_ours_bikes" --gen_file_path "./images/bg_ours/bikes/gen" --ref_file_path "./images/bg_ours/bikes/ref"
python evaluate.py --exp_name "bg_ours_interactive_bikes" --gen_file_path "./images/bg_ours_interactive/bikes/gen" --ref_file_path "./images/bg_ours_interactive/bikes/ref"
python evaluate.py --exp_name "bg_replaceanything_bikes" --gen_file_path "./images/bg_replaceanything/bikes/gen" --ref_file_path "./images/bg_replaceanything/bikes/ref"
python evaluate.py --exp_name "bg_shopify_bikes" --gen_file_path "./images/bg_shopify/bikes/gen" --ref_file_path "./images/bg_shopify/bikes/ref"

python evaluate.py --exp_name "bg_ours_cars" --gen_file_path "./images/bg_ours/cars/gen" --ref_file_path "./images/bg_ours/cars/ref"
python evaluate.py --exp_name "bg_ours_interactive_cars" --gen_file_path "./images/bg_ours_interactive/cars/gen" --ref_file_path "./images/bg_ours_interactive/cars/ref"
python evaluate.py --exp_name "bg_replaceanything_cars" --gen_file_path "./images/bg_replaceanything/cars/gen" --ref_file_path "./images/bg_replaceanything/cars/ref"
python evaluate.py --exp_name "bg_shopify_car" --gen_file_path "./images/bg_shopify/cars/gen" --ref_file_path "./images/bg_shopify/cars/ref"

python evaluate.py --exp_name "bg_ours_products" --gen_file_path "./images/bg_ours/products/gen" --ref_file_path "./images/bg_ours/products/ref"
python evaluate.py --exp_name "bg_ours_interactive_products" --gen_file_path "./images/bg_ours_interactive/products/gen" --ref_file_path "./images/bg_ours_interactive/products/ref"
python evaluate.py --exp_name "bg_replaceanything_products" --gen_file_path "./images/bg_replaceanything/products/gen" --ref_file_path "./images/bg_replaceanything/products/ref"
python evaluate.py --exp_name "bg_shopify_products" --gen_file_path "./images/bg_shopify/products/gen" --ref_file_path "./images/bg_shopify/products/ref"

python evaluate.py --exp_name "bg_ours_overall" --gen_file_path "./images/bg_ours/overall/gen" --ref_file_path "./images/bg_ours/overall/ref"
python evaluate.py --exp_name "bg_ours_interactive_overall" --gen_file_path "./images/bg_ours_interactive/overall/gen" --ref_file_path "./images/bg_ours_interactive/overall/ref"
python evaluate.py --exp_name "bg_replaceanything_overall" --gen_file_path "./images/bg_replaceanything/overall/gen" --ref_file_path "./images/bg_replaceanything/overall/ref"
python evaluate.py --exp_name "bg_shopify_overall" --gen_file_path "./images/bg_shopify/overall/gen" --ref_file_path "./images/bg_shopify/overall/ref"

python evaluate.py --exp_name "comp_ours_bikes" --gen_file_path "./images/comp_ours/bikes/gen" --ref_file_path "./images/comp_ours/bikes/ref" --masks_path "./images/comp_ours/bikes/masks"
python evaluate.py --exp_name "comp_tficon_bikes" --gen_file_path "./images/comp_TF_Icon/bikes/gen" --ref_file_path "./images/comp_TF_Icon/bikes/ref" --masks_path "./images/comp_TF_Icon/bikes/masks"
python evaluate.py --exp_name "comp_anydoor_bikes" --gen_file_path "./images/comp_AnyDoor/bikes/gen" --ref_file_path "./images/comp_AnyDoor/bikes/ref" --masks_path "./images/comp_AnyDoor/bikes/masks"

python evaluate.py --exp_name "comp_ours_cars" --gen_file_path "./images/comp_ours/cars/gen" --ref_file_path "./images/comp_ours/cars/ref" --masks_path "./images/comp_ours/cars/masks"
python evaluate.py --exp_name "comp_tficon_cars" --gen_file_path "./images/comp_TF_Icon/cars/gen" --ref_file_path "./images/comp_TF_Icon/cars/ref" --masks_path "./images/comp_TF_Icon/cars/masks"
python evaluate.py --exp_name "comp_anydoor_cars" --gen_file_path "./images/comp_AnyDoor/cars/gen" --ref_file_path "./images/comp_AnyDoor/cars/ref" --masks_path "./images/comp_AnyDoor/cars/masks"

python evaluate.py --exp_name "comp_ours_products" --gen_file_path "./images/comp_ours/products/gen" --ref_file_path "./images/comp_ours/products/ref" --masks_path "./images/comp_ours/products/masks"
python evaluate.py --exp_name "comp_tficon_products" --gen_file_path "./images/comp_TF_Icon/products/gen" --ref_file_path "./images/comp_TF_Icon/products/ref" --masks_path "./images/comp_TF_Icon/products/masks"
python evaluate.py --exp_name "comp_anydoor_products" --gen_file_path "./images/comp_AnyDoor/products/gen" --ref_file_path "./images/comp_AnyDoor/products/ref" --masks_path "./images/comp_AnyDoor/products/masks"

python evaluate.py --exp_name "comp_ours_overall" --gen_file_path "./images/comp_ours/overall/gen" --ref_file_path "./images/comp_ours/overall/ref" --masks_path "./images/comp_ours/overall/masks"
python evaluate.py --exp_name "comp_tficon_overall" --gen_file_path "./images/comp_TF_Icon/overall/gen" --ref_file_path "./images/comp_TF_Icon/overall/ref" --masks_path "./images/comp_TF_Icon/overall/masks"
python evaluate.py --exp_name "comp_anydoor_overall" --gen_file_path "./images/comp_AnyDoor/overall/gen" --ref_file_path "./images/comp_AnyDoor/overall/ref" --masks_path "./images/comp_AnyDoor/overall/masks"

# TODO: add human eval
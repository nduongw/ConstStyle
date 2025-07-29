
# for domain in "sketch"; do
#     for class in "dog"; do
#         CUDA_VISIBLE_DEVICES=0 python gen_images.py  \
#         --content_dir ../DATA/pacs/images/${domain}/${class} \
#         --style ../DATA/pacs/images/art_painting/ \
#         --store_folder ./styled_images_ver2_3domains/pacs/images/${domain}/${class} \
#         --selected_domain art_painting
#     done
# done

CUDA_VISIBLE_DEVICES=0 python all_seen_domain.py --root ./styled_images_ver2_3domains --dataset-config-file ../configs/datasets/dg/pacs_total.yaml  --resume false
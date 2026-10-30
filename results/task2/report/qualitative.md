# Qualitative examples (196 visual tokens; seeded selection, seed 6304)


## Pair solved, swap flips the answer to the partner answer - grounding_test (64 of 128 questions)

| image id | question | reference | partner ref. | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|---|---|
| 189550 | Does this image contain a truck? | no | yes | 'no' | 'yes' | 'no' | 'No, the image does not contain a truck.' |
| 80213 | Does this image contain a motorcycle? | no | yes | 'no' | 'yes' | 'no' | 'No, the image does not contain a motorcycle.' |
| 231691 | Does this image contain a wine glass? | no | yes | 'no' | 'yes' | 'no' | 'No, the image does not contain a wine glass.' |
| 476335 | Does this image contain a refrigerator? | yes | no | 'yes' | 'no' | 'no' | 'No, the image does not contain a refrigerator.' |

## Pair solved, swap flips the answer to the partner answer - vqa_test (54 of 192 questions)

| image id | question | reference | partner ref. | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|---|---|
| 250309 | What sport is being played? | tennis | soccer | 'tennis' | 'soccer' | 'none' | 'The sport being played is **basketball**.' |
| 13175 | Is the man on a skateboard? | yes | no | 'yes' | 'no' | 'no' | 'No, the man is not on a skateboard.' |
| 392905 | Is the elephant confined? | yes | no | 'yes' | 'no' | 'no' | 'No, the elephant is not confined.' |
| 331250 | How many people are in the picture? | 3 | 1 | '3' | '1' | '1' | 'The number of people in the picture is **not provided**. Please provide more' |

## Correct with the right image but unchanged under the complementary image - grounding_test (12 of 128 questions)

| image id | question | reference | partner ref. | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|---|---|
| 247790 | How many dining tables can you see? | 3 | 1 | '3' | '3' | '3' | 'The number of dining tables you can see depends on the number of tables you are' |
| 92244 | How many parking meters can you see? | 2 | 3 | '2' | '2' | '1' | 'It depends on the location and the type of parking meter. In many urban areas' |
| 191949 | Does this image contain a book? | no | yes | 'no' | 'no' | 'no' | 'No, the image does not contain a book.' |
| 189752 | Does this image contain a chair? | yes | no | 'yes' | 'yes' | 'no' | 'No, the image does not contain a chair.' |

## Correct with the right image but unchanged under the complementary image - vqa_test (33 of 192 questions)

| image id | question | reference | partner ref. | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|---|---|
| 223090 | Are the carrots whole? | yes | no | 'yes' | 'yes' | 'no' | 'No, carrots are not whole. They are typically cut into pieces or chopped for' |
| 314880 | Which color team has more visible players? | red | black | 'red' | 'red' | 'blue' | 'The color team with more visible players depends on the specific context, such as a' |
| 137954 | Is the boys hair curly? | yes | no | 'yes' | 'yes' | 'no' | 'No, the boys hair is not curly.' |
| 44612 | Was the photographer standing on dry ground when this photo was taken? | yes | no | 'yes' | 'yes' | 'yes' | 'The photographer was standing on **dry ground** when the photo was taken.' |

## Wrong with the right image, answer equals the partner answer (language prior / insensitivity) - grounding_test (18 of 128 questions)

| image id | question | reference | partner ref. | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|---|---|
| 41104 | How many horses can you see? | 4 | 3 | '3' | '2' | '1' | 'I cannot see horses, but I can see horses in the wild.' |
| 575931 | Does this image contain a car? | yes | no | 'no' | 'no' | 'no' | 'No, the image does not contain a car.' |
| 308430 | How many dining tables can you see? | 1 | 3 | '3' | '3' | '3' | 'The number of dining tables you can see depends on the number of tables you are' |
| 93068 | Does this image contain a suitcase? | no | yes | 'yes' | 'yes' | 'yes' | 'No, the image does not contain a suitcase.' |

## Wrong with the right image, answer equals the partner answer (language prior / insensitivity) - vqa_test (44 of 192 questions)

| image id | question | reference | partner ref. | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|---|---|
| 568265 | What color tips does the kite have? | red | blue | 'blue' | 'blue' | 'red' | 'The color of the kite is typically **blue**.' |
| 551338 | Is the girl playing tennis? | no | yes | 'yes' | 'yes' | 'no' | 'No, the girl is not playing tennis.' |
| 499672 | What color is the bridle? | red | black | 'black' | 'black' | 'white' | 'The color of the bridle is **black**.' |
| 293811 | What's the color of the batter's helmet? | blue | red | 'red' | 'red' | 'black' | "The color of the batter's helmet is typically **black**." |

## Wrong with the right image, other error - grounding_test (24 of 128 questions)

| image id | question | reference | partner ref. | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|---|---|
| 452137 | How many hot dogs can you see? | 4 | 2 | '1' | '3' | '2' | 'There are no hot dogs visible in the sky.' |
| 363100 | How many clocks can you see? | 2 | 4 | '1' | '4' | '1' | 'The number of clocks you can see depends on the context and the specific location.' |
| 302842 | How many people can you see? | 1 | 4 | '2' | '3' | '2' | 'It depends on the location and time of observation. In urban areas, you might' |
| 47367 | How many fire hydrants can you see? | 2 | 4 | '1' | '4' | '1' | 'There is no specific number of fire hydrants that can be seen in a single' |

## Wrong with the right image, other error - vqa_test (46 of 192 questions)

| image id | question | reference | partner ref. | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|---|---|
| 112342 | What animal is here besides the elephants? | dog | ducks | 'elephant' | 'elephants' | 'elephants' | 'The animal here besides elephants is **tigers**.' |
| 551910 | What is he doing? | flying kite | skateboarding | 'kicking' | 'skating' | 'taking the bus' | "I don't have enough information to answer that question. Could you please provide more" |
| 435466 | How many chairs are stacked? | 0 | 2 | '1' | '1' | '1' | 'The question "How many chairs are stacked?" is ambiguous and unclear. It could' |
| 317969 | What is crossing the road? | sheep | cow | 'bus' | 'horse' | 'elephant' | 'Crossing the road refers to the act of moving from one road to another,' |

# Aligned complementary pairs

Rule: for each suite, the first 3 pairs (pair index order) in each category, decided by exact canonical match under the original image. Each row is one member; both members share the question.


## grounding_test - both correct (32 of 64 pairs)

| pair | question | image id | reference | pred. (own image) | pred. (partner image) | pred. (mismatched) | pred. (no image) |
|---|---|---|---|---|---|---|---|
| val_pres_0 | Does this image contain a parking meter? | 174019 | yes | 'yes' | 'no' | 'no' | 'No, the image does not contain a parking meter.' |
| val_pres_0 | Does this image contain a parking meter? | 163589 | no | 'no' | 'yes' | 'no' | 'No, the image does not contain a parking meter.' |
| val_pres_1 | Does this image contain a truck? | 339870 | yes | 'yes' | 'no' | 'no' | 'No, the image does not contain a truck.' |
| val_pres_1 | Does this image contain a truck? | 189550 | no | 'no' | 'yes' | 'no' | 'No, the image does not contain a truck.' |
| val_pres_3 | Does this image contain a carrot? | 251537 | yes | 'yes' | 'no' | 'no' | 'No, the image does not contain a carrot.' |
| val_pres_3 | Does this image contain a carrot? | 535579 | no | 'no' | 'yes' | 'no' | 'No, the image does not contain a carrot.' |

## grounding_test - one correct (22 of 64 pairs)

| pair | question | image id | reference | pred. (own image) | pred. (partner image) | pred. (mismatched) | pred. (no image) |
|---|---|---|---|---|---|---|---|
| val_pres_2 | Does this image contain a book? | 373440 | yes | 'no' | 'no' | 'no' | 'No, the image does not contain a book.' |
| val_pres_2 | Does this image contain a book? | 191949 | no | 'no' | 'no' | 'no' | 'No, the image does not contain a book.' |
| val_pres_12 | Does this image contain a suitcase? | 388569 | yes | 'yes' | 'yes' | 'yes' | 'No, the image does not contain a suitcase.' |
| val_pres_12 | Does this image contain a suitcase? | 93068 | no | 'yes' | 'yes' | 'yes' | 'No, the image does not contain a suitcase.' |
| val_pres_15 | Does this image contain a handbag? | 155012 | yes | 'no' | 'no' | 'no' | 'No, the image does not contain a handbag.' |
| val_pres_15 | Does this image contain a handbag? | 57725 | no | 'no' | 'no' | 'no' | 'No, the image does not contain a handbag.' |

## grounding_test - both wrong (10 of 64 pairs)

| pair | question | image id | reference | pred. (own image) | pred. (partner image) | pred. (mismatched) | pred. (no image) |
|---|---|---|---|---|---|---|---|
| val_count_0 | How many people can you see? | 302842 | 1 | '2' | '3' | '2' | 'It depends on the location and time of observation. In urba |
| val_count_0 | How many people can you see? | 53949 | 4 | '3' | '2' | '1' | 'It depends on the location and time of observation. In urba |
| val_count_1 | How many hot dogs can you see? | 452137 | 4 | '1' | '3' | '2' | 'There are no hot dogs visible in the sky.' |
| val_count_1 | How many hot dogs can you see? | 179898 | 2 | '3' | '1' | '1' | 'There are no hot dogs visible in the sky.' |
| val_count_4 | How many refrigerators can you see? | 57936 | 2 | '1' | '2' | '1' | '0' |
| val_count_4 | How many refrigerators can you see? | 180274 | 4 | '2' | '1' | '1' | '0' |

## vqa_test - both correct (27 of 96 pairs)

| pair | question | image id | reference | pred. (own image) | pred. (partner image) | pred. (mismatched) | pred. (no image) |
|---|---|---|---|---|---|---|---|
| val_13175000_314356022 | Is the man on a skateboard? | 13175 | yes | 'yes' | 'no' | 'no' | 'No, the man is not on a skateboard.' |
| val_13175000_314356022 | Is the man on a skateboard? | 314356 | no | 'no' | 'yes' | 'no' | 'No, the man is not on a skateboard.' |
| val_450324006_161185003 | Is the light green? | 450324 | yes | 'yes' | 'no' | 'no' | "No, the light green is not the color you're referring to." |
| val_450324006_161185003 | Is the light green? | 161185 | no | 'no' | 'yes' | 'no' | "No, the light green is not the color you're referring to." |
| val_29504002_136117003 | Are these meals healthy? | 29504 | yes | 'yes' | 'no' | 'no' | "It depends on the specific meal you're referring to. Some m |
| val_29504002_136117003 | Are these meals healthy? | 136117 | no | 'no' | 'yes' | 'no' | "It depends on the specific meal you're referring to. Some m |

## vqa_test - one correct (48 of 96 pairs)

| pair | question | image id | reference | pred. (own image) | pred. (partner image) | pred. (mismatched) | pred. (no image) |
|---|---|---|---|---|---|---|---|
| val_7873000_578492014 | Is the woman wearing a hat? | 7873 | no | 'yes' | 'yes' | 'no' | 'No, the woman is not wearing a hat.' |
| val_7873000_578492014 | Is the woman wearing a hat? | 578492 | yes | 'yes' | 'yes' | 'no' | 'No, the woman is not wearing a hat.' |
| val_137954003_369404004 | Is the boys hair curly? | 137954 | yes | 'yes' | 'yes' | 'no' | 'No, the boys hair is not curly.' |
| val_137954003_369404004 | Is the boys hair curly? | 369404 | no | 'yes' | 'yes' | 'no' | 'No, the boys hair is not curly.' |
| val_162818002_296775002 | Is the bus moving? | 162818 | no | 'yes' | 'yes' | 'no' | 'No, the bus is not moving.' |
| val_162818002_296775002 | Is the bus moving? | 296775 | yes | 'yes' | 'yes' | 'no' | 'No, the bus is not moving.' |

## vqa_test - both wrong (21 of 96 pairs)

| pair | question | image id | reference | pred. (own image) | pred. (partner image) | pred. (mismatched) | pred. (no image) |
|---|---|---|---|---|---|---|---|
| val_95805000_546792001 | Is this real? | 95805 | no | 'yes' | 'no' | 'yes' | 'Yes, this is real.' |
| val_95805000_546792001 | Is this real? | 546792 | yes | 'no' | 'yes' | 'no' | 'Yes, this is real.' |
| val_21864004_156276000 | Is there an island in the picture? | 21864 | yes | 'no' | 'yes' | 'no' | 'Yes, there is an island in the picture.' |
| val_21864004_156276000 | Is there an island in the picture? | 156276 | no | 'yes' | 'no' | 'no' | 'Yes, there is an island in the picture.' |
| val_31620002_47627000 | Is her dress long? | 31620 | yes | 'no' | 'yes' | 'no' | 'No, her dress is not long.' |
| val_31620002_47627000 | Is her dress long? | 47627 | no | 'yes' | 'no' | 'yes' | 'No, her dress is not long.' |

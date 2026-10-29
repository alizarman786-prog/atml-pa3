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

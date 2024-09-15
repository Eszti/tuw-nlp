# Evaluation
## Test results

|  model | predicted extractions | gold extractions | matches | exact matches | prec | rec | F1 |
|-------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | 
| gr100_k2 | 4064 | 5372 | 935 | 11 | 0.1689 | 0.0946 | 0.1213 | 
| random_k10 | 5358 | 5372 | 1293 | 27 | 0.1596 | 0.1252 | 0.1403 | 
| ls_oie | 5805 | 5372 | 2383 | 852 | 0.3419 | 0.3541 | 0.3479 | 
| ls_oie_crf | 5723 | 5372 | 2478 | 937 | 0.3640 | 0.3607 | 0.3623 | 
| srl_bert | 5837 | 5372 | 2536 | 1042 | **0.3714** | **0.3819** | **0.3766** | 
| srl_bert_oie2016 | 5609 | 5372 | 2351 | 687 | 0.3565 | 0.3306 | 0.3430 | 


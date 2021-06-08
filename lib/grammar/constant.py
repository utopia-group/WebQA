ENTITY = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "LAW", "LANGUAGE", "DATE",
          "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "NOUN"]

PRED_ENTITY = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "LAW", "LANGUAGE", "DATE",
               "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "NOUN"]

THRESHOLD = [1.0, 0.95, 0.9, 0.85, 0.8]
THRESHOLD_2 = [1.0, 0.95, 0.9, 0.85, 0.8, 0.0]
THRESHOLD_3 = [1.0]
K = [1, 2, 3]


def update_entity(new_list):
    global ENTITY
    ENTITY = new_list


def get_entity():
    global ENTITY
    return ENTITY


def update_pred_entity(new_list):
    global PRED_ENTITY
    PRED_ENTITY = new_list


def get_pred_entity():
    global PRED_ENTITY
    return PRED_ENTITY

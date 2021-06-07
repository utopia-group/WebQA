import time, datetime
from lib.nlp.qa_bert.run_squad_new import initialize, evaluate
from lib.nlp.qa_bert.squad_generator import convert_context_and_questions_to_squad

class RunBertQA():
    def __init__(self, new_args=None):
        self.args, self.model, self.tokenizer = initialize(new_args)

    def predict_from_input_squad(self, contexts, questions, _id, get_mapping=False):
        squad_dict = convert_context_and_questions_to_squad(contexts, questions)
        return self.package_squad_prediction(squad_dict, _id, get_mapping=get_mapping)

    def package_squad_prediction(self, squad_dict, id="context-default", get_mapping=False):
        (section_predicition, prediction), dt = self.evaluate_input(squad_dict, get_mapping=get_mapping)
        packaged_predictions = []
        for entry in squad_dict["data"]:
            title = entry["title"]
            inner_package = []
            for p in entry["paragraphs"]:
                context = p["context"]
                qas = [(q["question"], prediction[q["id"]],
                        datetime.datetime.now().strftime("%d %B %Y %I:%M%p"),
                        "%0.02f seconds" % (dt),
                        '#' + id) for q in p["qas"]]
                inner_package.append((context, qas))
            packaged_predictions.append((title, inner_package))
        return section_predicition, packaged_predictions

    def generate_highlight(self, context, id, start_index, stop_index):
        if start_index > -1:
            context_split = context.split()
            start_index = len(" ".join(context_split[:start_index]))
            stop_index = len(" ".join(context_split[:stop_index + 1]))
        return context[(start_index+1):stop_index]
        # return 'highlight(' + '"#' + id + '",' + str(start_index) + ',' + str(stop_index) + ');return false;'

    def evaluate_input(self, squad_dict, passthrough=False, get_mapping=False):
        self.args.input_data = squad_dict
        t = time.time()
        predictions = evaluate(self.args, self.model, self.tokenizer, get_mapping=get_mapping)
        dt = time.time() - t
        print("Loading time: %0.02f seconds" % (dt))
        if passthrough:
            return predictions, squad_dict, dt
        return predictions, dt

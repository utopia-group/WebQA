"""
Configuration options for running tasks.
"""
import lib.spec as spec
from lib.program import TopLevelProgram

test_benchmarks = {
    'test': [
        'test_1',
        'test_2',
        'test_3',
        'test_4',
        'test_5',
    ],
    'fac': [
        'fac_1',
        'fac_2',
        'fac_3',
        'fac_4',
        'fac_5',
        'fac_6',
        'fac_7',
        'fac_10',
        'fac_11',
        'fac_12',
        'fac_13',
        'fac_14',
        'fac_15',
        'fac_16',
        'fac_17',
        'fac_18',
        'fac_19',
        'fac_20',
        'fac_21',
        'fac_22',
        'fac_23',
        'fac_24',
        'fac_25',
        'fac_26',
        'fac_27',
        'fac_28',
        'fac_29',
        'fac_30',
        'fac_31',
        'fac_32',
        'fac_33',
        'fac_34',
        'fac_35',
        'fac_36',
        'fac_37',
        'fac_38',
        'fac_39',
        'fac_40',
        'fac_41',
        'fac_42',
        'fac_43',
        'fac_44',
        'fac_45'
    ],
    'conf': [
        'conf_1',
        'conf_2',
        'conf_3',
        'conf_4',
        'conf_5',
        'conf_6',
        'conf_7',
        'conf_8',
        'conf_9',
        'conf_10',
        'conf_11',
        'conf_12',
        'conf_13',
        'conf_14',
        'conf_15',
        'conf_16',
        'conf_17',
        'conf_18',
        'conf_19',
        'conf_20',
        'conf_21',
        'conf_23',
        'conf_24',
        'conf_25',
        'conf_26',
        'conf_27',
        'conf_28',
        'conf_29',
        'conf_30',
        'conf_31',
        'conf_32',
        'conf_33',
        'conf_34',
        'conf_35',
        'conf_36',
        'conf_37',
        'conf_38',
        'conf_39',
        'conf_40'
    ],
    'class': [
        'class_1',
        'class_2',
        'class_3',
        'class_4',
        'class_5',
        'class_6',
        'class_7',
        'class_8',
        'class_9',
        'class_10',
        'class_11',
        'class_12',
        'class_13',
        'class_15',
        'class_16',
        'class_17',
        'class_18',
        'class_19',
        'class_20',
        'class_21',
        'class_22',
        'class_23',
        'class_24',
        'class_25',
        'class_26',
        'class_27',
        'class_28',
        'class_29',
        'class_30',
        'class_31',
        'class_32',
        'class_33',
        'class_34',
        'class_35',
        'class_37',
        'class_38',
        'class_39',
        'class_40' 
    ],
    'clinic':[
        'clinic_1',
        'clinic_2',
        'clinic_3',
        'clinic_4',
        'clinic_5',
        'clinic_6',
        'clinic_7',
        'clinic_8',
        'clinic_9',
        'clinic_10',
        'clinic_11',
        'clinic_12',
        'clinic_13',
        'clinic_14',
        'clinic_15',
        'clinic_16',
        'clinic_17',
        'clinic_18',
        'clinic_19',
        'clinic_20',
        'clinic_21',
        'clinic_22',
        'clinic_23',
        'clinic_24',
        'clinic_25',
        'clinic_26',
        'clinic_27',
        'clinic_28',
        'clinic_29',
        'clinic_30',
        'clinic_31',
        'clinic_33',
        'clinic_34',
        'clinic_35',
        'clinic_36',
        'clinic_38',
        'clinic_39',
        'clinic_40',
        'clinic_41',
        'clinic_42',
    ]
}

# task training sets
task_train_set = {
    'test': {
        't1': ['1']
    },
    'fac': {
        't1': ['4', '5', '14', '26', '34'],
        't2': ['2', '11', '17', '32', '44'],
        't3': ['3', '12', '14', '36'],
        't4': ['3', '6', '25', '44'],
        't5': ['10', '17', '19', '22', '29'],
        't6': ['2', '22', '38', '41', '44'],
        't7': ['2', '19', '29', '41', '44'],
        't8': ['1', '31', '41', '42'],
    },
    'conf': {
        't1': ['1', '2', '13', '27', '37'],        
        't2': ['1', '5', '18', '30'],
        't3': ['8', '12', '20', '21'],
        't4': ['2', '7', '14', '18', '27'],
        't5': ['1', '9', '25', '37', '27'],
        't6': ['1', '5', '7', '30', '34'],
    },
    'class': {
        't1': ['1', '2', '4', '10', '31'],
        't2': ['1', '6', '10', '20'],
        't3': ['1', '4', '17', '25', '37'],
        't4': ['1', '8', '20', '26'],
        't5': ['3', '7', '13', '28', '33'],
        't6': ['1', '21', '22', '24', '30'],        
    },
    'clinic': {
        't1': ['2', '9', '17', '38', '40'],
        't2': ['7', '6', '9', '14', '39'],
        't3': ['8', '18', '27', '38'],        
        't4': ['1', '5', '12', '14', '33'],
        't5': ['1', '5', '10', '15', '16'],
    }
}

# List of tasks
tasks = {
    'test': {
        't1': spec.Task(
            'Who are the current PhD students?',
            ['Current Students', 'PhD'],
            ['PhD']
        )
    },
    'fac': {
        't1': spec.Task(
            'Who are the current PhD students?',
            ['Current Students', 'PhD'],
            ['PhD']
        ),
        't2': spec.Task(
            'What are the conference publications at PLDI?',
            ['Conference Publications'],
            ['PLDI']
        ),
        't3': spec.Task(
            'What courses does this person teach?',
            ['Courses', 'Teaching'],
            ['Courses', 'Teaching']
        ),
        't4': spec.Task(
            'What is the title of the paper that '
            'received the Best Paper Award?',
            ['Conference Publications'],
            ['Best Paper Award']
        ),
        't5': spec.Task(
            'What program committees or PC have this person served for?',
            ['Program Committee', 'PC'],
            ['PC', 'Program Committee']
        ),
        't6': spec.Task(
            'What conference papers have been published in 2012?',
            ['Conference Publications'],
            ['2012']
        ),
        't7': spec.Task(
            'Who are the co-authors among all papers published at PLDI?',
            ['Conference Publications'],
            ['PLDI']
        ),
        't8': spec.Task(
            'Who are the alumni or formerly advised students?',
            ['Alumni', 'Former Students'],
            ['Graduated']
        ),
    },
    'conf': {
        't1': spec.Task(
            'Who are the program co-chairs?',
            ['Program Chair', 'PC Chair', 'Programs Co-chair'],
            ['Chair', ' Co-chair', 'Chairs', 'Co-chairs']
        ),
        't2': spec.Task(
            'Who are the program committee (PC) members?',
            ['Program Committee', 'PC'],
            ['Program Committee', 'PC']
        ),
        't3': spec.Task(
            'What are the topics of interest?',
            ['Topics'],
            ['Topics']
        ),
        't4': spec.Task(
            'What is the paper submission deadline?',
            ['paper submission deadline'],
            ['paper submission', 'paper submissions', 'paper submission deadline']
        ),
        't5': spec.Task(
            'Is this conference double-blind or single-blind?',
            ['double-blind', 'single-blind'],
            ['double-blind', 'single-blind']
        ),
        't6': spec.Task(
            'What institutions are the program committee or PC members from?',
            ['PC', 'Program Committee'],
            ['PC', 'Program Committee']
        )
    },
    'class': {
        't1': spec.Task(
            'When are the lectures / sections?',
            ['Section', 'Lecture'],
            ['Section', 'Lecture']
        ),
        't2': spec.Task(
            'Who are the instructors?',
            ['Instructors'],
            ['Instructors']
        ),
        't3': spec.Task(
            'Who are the Teaching Assistants (TAs)?',
            ['Teaching Assistants', 'TAs'],
            ['Teaching Assistants', 'TAs', 'TA']
        ),
        't4': spec.Task(
            'When are the midterms or exams?',
            ['Exam', 'Midterm'],
            ['Exam', 'Midterm', 'Test']
        ),
        't5': spec.Task(
            'What are the textbooks?',
            ['Textbooks', 'Materials', 'Required Texts'],
            ['Textbooks', 'Materials', 'Required Texts', 'Texts'],
        ),
        't6': spec.Task(
            'How are the grades counted in this class?',
            ['grades', 'grading', 'rubric'],
            ['grades', 'grading', 'rubric']
        ),
    },
    'clinic': {
        't1': spec.Task(
            'Who are the doctors or providers?',
            ['Doctor', 'Provider', 'Our Team'],
            ['Doctor', 'Provider', 'Our Team']
        ),
        't2': spec.Task(
            'What types of service do they provide?',
            ['Our Services'],
            ['Services', 'Our Services'],
        ),
        't3': spec.Task(
            'What types of treatment do they specialize in?',
            ['Treatments', 'Specialties'],
            ['Treat', 'Treatments', 'Treated', 'Specialties'],
        ),
        't4': spec.Task(
            'What insurance plan do they accept?',
            ['Insurance', 'Plans Accepted'],
            ['Insurance']
        ),
        't5': spec.Task(
            # 'What is the address of the clinic?',
            'Where are the clinics located?',
            ['Location'],
            ['Location']
        )

    }
}


MANUAL_OVERALL_PROGRAM = {'fac': {
    't1':
        TopLevelProgram(init=[
            ('lambda: dsl.AnySat(dsl.GetChildren(dsl.GetNode(dsl.matchSection1), dsl.isAny), lambda x: dsl._true(x))',
             'lambda: dsl.GetEntity(dsl.ExtractContent(x), "PERSON")')])
    },
    'conf': {
        't1':
            TopLevelProgram(init=[(
                'lambda: dsl.AnySat(dsl.GetLeaves(dsl.GetNode(dsl.matchSection2, w, 1, 0.9), dsl.isAny), lambda x: dsl.hasHeader(x, 0.95))',
                'lambda: dsl.GetEntity(dsl.ExtractContent(x), "PERSON")'
            ),
            (
                'lambda: dsl.AnySat(dsl.GetLeaves(dsl.GetNode(dsl.matchSection2, w, 2, 0.85), '
                'dsl.isAny), lambda x: dsl._true(x))',
                'lambda: dsl.GetAnswer(dsl.Filter(dsl.ExtractContent(x), lambda x: dsl.hasString(x, const_str, '
                '0.85)), q, 1)'
            )
            ]),
    },
    'class': {
        't1': TopLevelProgram(init=[
            (
                'lambda: AnySat(dsl.GetChildren(dsl.GetNode(dsl.matchSection2, w, 3, 0.85), dsl.isAny),lambda x: dsl._true(x))',
                "lambda: dsl.GetEntity(dsl.ExtractContent(x), 'TIME')"
            )
        ])
    },
    'clinic': {
        't2': TopLevelProgram(init=[(
            'lambda: AnySat(dsl.GetLeaves(dsl.GetNode(dsl.matchKeyword, w, 1, 0.9), dsl.isAny),lambda x: dsl._true(x))',
            'lambda: dsl.Filter(dsl.Split(dsl.ExtractContent(x)), lambda x: dsl.hasEntity(x, "NOUN"))'
        )
        ])
    }
}

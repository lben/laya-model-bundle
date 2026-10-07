import unittest
from unittest.mock import Mock, patch

from laya_chunking import TextPlan, chunk_settings, predict_text


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.questions = {'bucket': {'type': 'choice', 'criteria': {'ordinary': 'ordinary', 'rare': 'rare'}}}
        self.agent = Mock()
        self.agent.tok.decode.side_effect = lambda ids, **kwargs: ''.join(ids)
        self.answer = {'answers': {'bucket': {'choice': 'ordinary', 'answer_confidence': 0.6,
                                              'probabilities': {'ordinary': 0.6, 'rare': 0.4}}},
                       'usage': {'input_tokens': 50, 'truncated': False, 'state_tokens_dropped': 0}}
        self.agent.predict.return_value = self.answer
        analysis_patch = patch('laya_chunking.analyze_text', side_effect=self.analyze)
        self.analysis = analysis_patch.start()
        self.addCleanup(analysis_patch.stop)

    def analyze(self, agent, text, questions, state_key, budgets):
        return TextPlan(list(text), len(text) + 7, 36, len(text) > 36, 64, 28)

    def test_short_auto_preserves_original_sdk_response(self):
        result = predict_text(self.agent, 'short', self.questions)
        self.assertIs(result, self.answer)
        self.agent.predict.assert_called_once_with({'message': 'short'}, self.questions)

    def test_auto_covers_every_token_and_keeps_every_option(self):
        text = 'abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJ'
        result = predict_text(self.agent, text, self.questions, chunk_overlap=4)
        calls = self.agent.predict.call_args_list
        covered = set()
        start = 0
        for call in calls:
            chunk = call.args[0]['message']
            self.assertEqual(call.args[1], self.questions)
            self.assertLessEqual(len(chunk), 20)
            self.assertEqual(chunk, text[start:start + len(chunk)])
            covered.update(range(start, start + len(chunk)))
            start += len(chunk) - 4
        self.assertEqual(covered, set(range(len(text))))
        self.assertEqual(result['chunking']['covered_tokens'], len(text))
        self.assertFalse(result['usage']['truncated'])
        self.assertEqual(result['usage']['state_tokens_dropped'], 0)

    def test_late_evidence_wins_without_averaging(self):
        text = 'neutral text ' * 5 + 'Z'
        def infer(state, questions, **kwargs):
            rare = 'Z' in state['body']
            probs = {'ordinary': 0.05, 'rare': 0.95} if rare else {'ordinary': 0.6, 'rare': 0.4}
            return {'answers': {'bucket': {'choice': 'rare' if rare else 'ordinary',
                                           'answer_confidence': max(probs.values()), 'probabilities': probs}},
                    'usage': {'truncated': False}}
        self.agent.predict.side_effect = infer
        result = predict_text(self.agent, text, self.questions, state_key='body', chunk_overlap=4)
        self.assertEqual(result['answers']['bucket']['choice'], 'rare')
        self.assertEqual(result['answers']['bucket']['probabilities']['rare'], 0.95)
        self.assertEqual(result['answers']['bucket']['window']['token_end'], len(text))

    def test_force_custom_size_and_no_overlap(self):
        result = predict_text(self.agent, 'abcdefghijklmnop', self.questions, chunking='on', chunk_tokens=6, chunk_overlap=0)
        self.assertEqual([call.args[0]['message'] for call in self.agent.predict.call_args_list], ['abcdef', 'ghijkl', 'mnop'])
        self.assertEqual(result['usage']['windows'], 3)

    def test_off_preserves_single_pass_and_marks_truncation(self):
        text = 'x' * 100
        result = predict_text(self.agent, text, self.questions, chunking='off', budgets={'max_len': 64})
        self.agent.predict.assert_called_once_with({'message': text}, self.questions, max_len=64)
        self.assertIn('first part', result['chunking']['warning'])

    def test_candidate_that_expands_is_shrunk_and_all_tokens_still_covered(self):
        def inspect(agent, text, questions, key, budgets):
            plan = self.analyze(agent, text, questions, key, budgets)
            if len(text) < 40 and len(text) > 12:
                plan.truncated = True
                plan.state_tokens = 42
            return plan
        self.analysis.side_effect = inspect
        predict_text(self.agent, 'x' * 45, self.questions, chunk_overlap=2)
        self.assertTrue(all(len(call.args[0]['message']) <= 12 for call in self.agent.predict.call_args_list))

    def test_no_partial_result_if_sdk_still_truncates_a_chunk(self):
        self.agent.predict.return_value = {'answers': {}, 'usage': {'truncated': True}}
        with self.assertRaisesRegex(RuntimeError, 'truncated a planned chunk'):
            predict_text(self.agent, 'x' * 50, self.questions)

    def test_invalid_controls_and_effective_overlap_fail(self):
        for values in [{'chunking': 'bad'}, {'chunk_tokens': 0}, {'chunk_overlap': -1},
                       {'chunk_tokens': 10, 'chunk_overlap': 10}, {'chunk_tokens': True}]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                chunk_settings(**values)
        with self.assertRaisesRegex(ValueError, 'effective chunk size'):
            predict_text(self.agent, 'x' * 50, self.questions, chunk_overlap=20)
        self.agent.predict.assert_not_called()

    def test_each_answer_keeps_its_own_deciding_chunk_and_original_result_is_unchanged(self):
        first = {'answers': {'choice': {'choice': 'A', 'answer_confidence': 0.9, 'probabilities': {'A': 0.9, 'B': 0.1}},
                             'yesno': {'noul': 0.1}, 'score': {'score': 0, 'answer_confidence': 0.2}}, 'usage': {'truncated': False}}
        second = {'answers': {'choice': {'choice': 'B', 'answer_confidence': 0.8, 'probabilities': {'A': 0.2, 'B': 0.8}},
                              'yesno': {'noul': 0.8}, 'score': {'score': 1, 'answer_confidence': 0.9}}, 'usage': {'truncated': False}}
        self.agent.predict.side_effect = [first, second]
        result = predict_text(self.agent, 'x' * 30, self.questions, chunking='on', chunk_tokens=15, chunk_overlap=0)
        self.assertEqual(result['answers']['choice']['window']['index'], 0)
        self.assertEqual(result['answers']['yesno']['window']['index'], 1)
        self.assertEqual(result['answers']['score']['window']['index'], 1)
        self.assertNotIn('window', first['answers']['choice'])
        self.assertNotIn('window', second['answers']['yesno'])


if __name__ == '__main__':
    unittest.main()

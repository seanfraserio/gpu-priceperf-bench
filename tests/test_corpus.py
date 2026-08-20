from gppb.corpus import build_corpus


def test_corpus_is_deterministic_across_calls():
    assert build_corpus(16, 1024) == build_corpus(16, 1024)


def test_corpus_prompts_are_mutually_distinct():
    prompts = build_corpus(64, 1024)
    assert len(set(prompts)) == 64


def test_corpus_length_is_within_two_percent_of_target():
    # Approximate token accounting: the harness records the server's reported
    # prompt_tokens, so we only need to be close, not exact.
    prompts = build_corpus(8, 1024)
    for p in prompts:
        approx_tokens = len(p.split())
        assert 1004 <= approx_tokens <= 1044


def test_changing_seed_changes_the_corpus():
    assert build_corpus(8, 1024, seed=1) != build_corpus(8, 1024, seed=2)

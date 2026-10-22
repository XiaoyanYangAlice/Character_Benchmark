"""测试用的假 OpenAI 兼容 client，供 test_eval_metrics.py 和
test_dialogue_runner.py 共用，避免同一份鸭子类型代码抄两遍。"""


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, responses):
        # responses: list[str]，每次 create() 调用按顺序弹出一个；用完了报错，
        # 这样测试能明确断言"调用了几次"，不会静默复用同一个响应掩盖 bug。
        self._responses = list(responses)

    def create(self, **kwargs):
        if not self._responses:
            raise AssertionError("被调用的次数超过了测试准备的响应数量")
        return _FakeResponse(self._responses.pop(0))


class _FakeChat:
    def __init__(self, responses):
        self.completions = _FakeCompletions(responses)


class FakeClient:
    """伪造 openai.OpenAI 的最小接口：client.chat.completions.create(...)。"""

    def __init__(self, responses):
        self.chat = _FakeChat(responses)

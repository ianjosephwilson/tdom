from .parser import parse_html
from .printer import pformat_tnode


def test_printing():
    attrs = dict(spread_attrs=True)
    sample_t = t"""
<!doctype html>
<html>
    <body>
        <div class="bg-red" class={"green"} {attrs}>
        This is sample {"text"}.<span>With tag</span>
        </div>
        <form>
            <input disabled>
        </form>
    </body>
</html>
"""
    exp = """<tnode-fragment>
  <!DOCTYPE html>
  <html>
    <body>
      <div (('class', 'bg-red'), ('class', 0), (1,))>
        <text>
        This is sample </text><slot>{2}</slot><text>.</text>
        <span>
          <text>With tag</text>
        </span>
      </div>
      <form>
        <input (('disabled',),)>
      </form>
    </body>
  </html>
</tnode-fragment>
"""
    res = "".join(pformat_tnode(parse_html(sample_t), show_text=True))
    assert res == exp

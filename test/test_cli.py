import json

import pytest

from galrax.padic.cli import main, parse_poly


def test_rec(capsys):
    pytest.importorskip("flint")
    assert main(["rec", "1,1", "0,1", "--value", "2", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["certified"] and out["zeros"] == [-3, 3]
    assert main(["rec", "1,1", "0,1", "--certificate"]) == 0
    assert "certified: n in Z with u_n = 0: {0}" in capsys.readouterr().out


def test_map_and_henon(capsys):
    assert main(["map", "--vars", "x,y", "y", "x*y + 1", "--start", "1,1", "--target", "7"]) == 0
    assert "{4}" in capsys.readouterr().out  # (1, 1), (1, 2), (2, 3), (3, 7), (7, 22)
    assert main(["map", "--vars", "i,x", "i + 1", "2*x", "--start", "0,1",
                 "--until", "x - 2*i - 22", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["zeros"] == [5]
    assert main(["henon", "--c", "1", "--P", "0,0,0", "--start", "0,1", "--target", "0"]) == 0
    assert "{0} + 2Z" in capsys.readouterr().out


def test_tpls(tmp_path, capsys):
    pytest.importorskip("flint")
    path = tmp_path / "three.tpls"
    path.write_text("(2, (1, 1), (2, 3))\n(2, (1, 2), (0, 1))\n(2, (1, 0), (1, 1))\n")
    assert main(["tpls", str(path), "--timeout", "60"]) == 3  # r_{k-1} = 0 in the last
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["zeros"] for r in rows[:2]] == [[-3], [0]]
    assert not rows[2]["certified"] and "error" in rows[2]


def test_expressions_are_whitelisted():
    assert parse_poly("(x - y)**2 + 3", ["x", "y"]) == {(2, 0): 1, (1, 1): -2, (0, 2): 1, (0, 0): 3}
    for bad in ("__import__('os')", "x.real", "x / 2", "x ** y", "z", "[x]"):
        with pytest.raises(ValueError):
            parse_poly(bad, ["x", "y"])
    with pytest.raises(SystemExit) as error:
        main(["map", "__import__('os')", "--start", "1", "--target", "0"])
    assert error.value.code == 2

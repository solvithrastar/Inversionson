import os, shutil
from pathlib import Path
from git import Git

def _test():

    script_dir = Path(os.path.dirname(os.path.realpath(__file__)))

    def rmIfExists(path, directory=False):
        path = Path(path)
        if not path.exists():
            return
        if directory:
            assert path.is_dir()
            shutil.rmtree(path)
        else:
            assert path.is_file()
            os.remove(path)


    repo = Path(script_dir / "test-repo")
    rmIfExists(repo, True)
    shutil.copytree(script_dir / "test_data", repo)

    git = Git(repo, author_name="test", author_email="<test@dummy-email.ch>")
    git.init()

    with open(repo / "a", "w") as f:
        f.write("line 0 in a\n")

    git.ignore("a")
    git.lfs_track("*.h5", "*.sqlite")
    git.add_commit("-A")
    git.branch("b")
    assert git.is_branch("main")
    assert git.is_branch("b")
    assert not git.is_branch("a")
    assert git.status("a") == "!!"

    with open(repo / "b", "w") as f:
        f.write("line 0 in b\n")

    git.add_commit("b")
    assert not git.status("b")
    with open(repo / "b", "w") as f:
        f.write("line 1 in b\n")

    assert git.status("b") == "M"
    git.run("reset", "--hard")
     
    git.branch("main")
    assert not (repo / "b").exists()


if __name__ == "__main__":
    _test()
    print("Test passed")

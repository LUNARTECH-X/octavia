import subprocess


def test_docker_build_and_run():
    print("Testing Docker build...")
    build = subprocess.run(["docker", "compose", "build"], capture_output=True, text=True)
    assert build.returncode == 0, "Docker build failed"
    print("Docker build successful")

    print("Testing Docker run...")
    run = subprocess.run(["docker", "compose", "up", "--abort-on-container-exit"], capture_output=True, text=True)
    assert run.returncode == 0, "Docker container failed to start"
    print("Container started")
    print(run.stdout)

if __name__ == "__main__":
    test_docker_build_and_run()

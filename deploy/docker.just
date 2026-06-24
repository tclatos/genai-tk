# Docker deployment recipes for genai-tk based applications.
#
# Import in your project justfile:
#   import 'deploy/docker.just'
#
# Required variables (define BEFORE the import):
#   app_name, docker_tag, extras, install_node, extra_copy,
#   streamlit_entry, docker_port, dockerfile, docker_pkg
#
# Example:
#   app_name     := "my-app"
#   docker_tag   := "latest"
#   extras       := "streamlit,nlp"
#   install_node := "false"
#   extra_copy   := "skills data"
#   streamlit_entry := ""
#   docker_port  := "8501"
#   dockerfile   := "deploy/Dockerfile"
#   docker_pkg   := "my_app"

# Full image reference (computed from parent variables)
_image := app_name + ":" + docker_tag

# ── Recipes ──────────────────────────────────────────────────────────────────

[doc('Build the Docker image')]
docker-build *args:
    docker build --pull --rm \
        -f "{{ dockerfile }}" \
        --build-arg PKG_NAME="{{ docker_pkg }}" \
        --build-arg EXTRAS="{{ extras }}" \
        --build-arg INSTALL_NODE="{{ install_node }}" \
        --build-arg STREAMLIT_ENTRY="{{ streamlit_entry }}" \
        --build-arg EXTRA_COPY_DIRS="{{ extra_copy }}" \
        -t {{ _image }} \
        {{ args }} \
        "."

[doc('Run the container (detached) with .env and optional data mount')]
docker-run *args:
    #!/usr/bin/env bash
    set -euo pipefail
    docker_args=(-d --name "{{ app_name }}" -p "{{ docker_port }}:8501")
    # Mount .env if present
    [[ -f .env ]] && docker_args+=(--env-file .env)
    # Mount ~/.env if present (genai-tk convention)
    [[ -f ~/.env ]] && docker_args+=(--env-file ~/.env)
    # Mount config override if DOCKER_CONFIG_DIR is set
    [[ -n "${DOCKER_CONFIG_DIR:-}" ]] && docker_args+=(-v "${DOCKER_CONFIG_DIR}:/app/config:ro")
    # Mount data volume if DOCKER_DATA_DIR is set
    [[ -n "${DOCKER_DATA_DIR:-}" ]] && docker_args+=(-v "${DOCKER_DATA_DIR}:/data/external")
    docker run "${docker_args[@]}" {{ args }} {{ _image }}
    echo "Container '{{ app_name }}' started on port {{ docker_port }}"

[doc('Run the container interactively (foreground, removed on exit)')]
docker-run-it *args:
    #!/usr/bin/env bash
    set -euo pipefail
    docker_args=(--rm -it --name "{{ app_name }}" -p "{{ docker_port }}:8501")
    [[ -f .env ]] && docker_args+=(--env-file .env)
    [[ -f ~/.env ]] && docker_args+=(--env-file ~/.env)
    [[ -n "${DOCKER_CONFIG_DIR:-}" ]] && docker_args+=(-v "${DOCKER_CONFIG_DIR}:/app/config:ro")
    [[ -n "${DOCKER_DATA_DIR:-}" ]] && docker_args+=(-v "${DOCKER_DATA_DIR}:/data/external")
    docker run "${docker_args[@]}" {{ args }} {{ _image }}

[doc('Stop and remove the container')]
docker-stop:
    -docker stop "{{ app_name }}" 2>/dev/null
    -docker rm "{{ app_name }}" 2>/dev/null

[doc('Show container logs (follow)')]
docker-logs *args:
    docker logs -f "{{ app_name }}" {{ args }}

[doc('Open a shell in the running container')]
docker-shell:
    docker exec -it "{{ app_name }}" bash

[doc('Check if the image is built')]
docker-check:
    @docker images --format "table {{{{.Repository}}}}\t{{{{.Tag}}}}\t{{{{.Size}}}}\t{{{{.CreatedSince}}}}" \
        | head -1
    @docker images --format "table {{{{.Repository}}}}\t{{{{.Tag}}}}\t{{{{.Size}}}}\t{{{{.CreatedSince}}}}" \
        | grep "{{ app_name }}" || echo "No image found for {{ app_name }}"

[doc('Remove the Docker image')]
docker-rmi:
    docker rmi {{ _image }}

[doc('Sync time (needed for WSL after hibernation)')]
docker-sync-time:
    sudo hwclock -s

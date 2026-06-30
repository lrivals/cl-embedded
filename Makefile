# Root Makefile — entry points for the GitLab publication pipeline.
#
# The firmware build lives in firmware/stm32f4_blink/Makefile (unchanged).
# These targets wrap scripts/prepare_gitlab_release.py — see docs/gitlab_publication.md.

PYTHON ?= python

.PHONY: gitlab-release gitlab-release-dry gitlab-check help

help:
	@echo "Targets:"
	@echo "  gitlab-release-dry   Show the sanitization plan (no writes)"
	@echo "  gitlab-release       Build the sanitized GitLab snapshot (runs tests + trace gate)"
	@echo "  gitlab-check         Verify the export would stay trace-free (no commit)"

## Show what would be excluded / rewritten / generated, without writing anything.
gitlab-release-dry:
	$(PYTHON) scripts/prepare_gitlab_release.py --dry-run

## Build the sanitized export into a separate repo, gated by tests + trace scan.
## Push manually afterwards (see command printed at the end), or add `ARGS=--push`.
gitlab-release:
	$(PYTHON) scripts/prepare_gitlab_release.py --run-tests $(ARGS)

## Future-additions guard: build the export to a throwaway dir and run the gate.
gitlab-check:
	$(PYTHON) scripts/prepare_gitlab_release.py --check-only

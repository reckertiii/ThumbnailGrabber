# Third-party notices

Thumbnail Grabber is proprietary (see `LICENSE`), but it depends on third-party
packages that carry their own licences. Those licences govern those packages. Most are
permissive and impose no restriction on how this software is licensed. **One is not**,
and it is called out below.

Licences recorded from the installed distributions.

| Package | Licence | Permissive? |
|---|---|---|
| opencv-python | Apache-2.0 | yes |
| mediapipe | Apache-2.0 | yes |
| numpy | BSD-3-Clause | yes |
| pillow | MIT-CMU (HPND) | yes |
| scikit-learn | BSD-3-Clause | yes |
| fastapi | MIT | yes |
| uvicorn | BSD-3-Clause | yes |
| python-multipart | Apache-2.0 | yes |
| pydantic | MIT | yes |
| anthropic | MIT | yes |
| **PyQt6** | **GPL-3.0-only**, or a commercial licence from Riverbank | **no — see below** |

## PyQt6 and proprietary licensing

PyQt6 is dual-licensed: GPL v3, or a paid commercial licence from Riverbank Computing
that permits proprietary applications. Under the GPL option, distributing a program
that links PyQt6 generally requires the combined work to be offered under the GPL —
which is incompatible with the proprietary terms in `LICENSE`.

This affects **only the desktop app** (`gui.py`, and the GUI branch of `main.py`).

It does **not** affect:

- **The container image.** The Dockerfile filters PyQt6 out of the install, so the
  published image contains no GPL component. This is what runs on TrueNAS.
- **The web UI** (`web/server.py` and the browser front-end).
- **The headless CLI scanner** (`main.py --scan-video --headless`), which never imports
  PyQt6.
- **Running the desktop app yourself.** GPL obligations attach to distribution, not to
  private use.

If the desktop app is ever distributed to others, the options are: buy a PyQt
commercial licence, port `gui.py` to PySide6 (the Qt Company's own bindings, LGPL,
which permits proprietary use when dynamically linked — as Python bindings are), or
drop the desktop app and ship only the web UI.

This is a summary written for orientation, not legal advice.

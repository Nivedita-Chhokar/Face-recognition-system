# Face-recognition-system

Smart Society Security demo.

## Install on Apple Silicon macOS

The pinned `dlib` source build in `requirements.txt` may fail with recent Apple
toolchains. This project includes `requirements-macos-arm64.txt`, which uses a
prebuilt dlib wheel instead. Install its listed packages with `--no-deps` so pip
doesn't try to build the separate `dlib` dependency:

```bash
cd smart_society_security
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-deps -r requirements-macos-arm64.txt
```

## Run

Create the database:

```bash
python database_setup.py
```

Register residents one at a time (repeat for each resident). Follow the prompts,
then press `c` to capture a face sample; capture five samples for each resident.
Press `q` to exit:

```bash
python register_faces.py
```

On macOS, allow camera access for the terminal or app running Python in
**System Settings > Privacy & Security > Camera**.

Start the live webcam system and press `q` to stop:

```bash
python security_system.py
```

For optional image evaluation, create `test_images/<person_name>/` folders with
test photos, then run:

```bash
python evaluate_system.py
```

Use this as a demo, not as the sole basis for granting or denying entry. Get
residents' informed consent and protect the face images, encodings, and access
logs as sensitive personal data.
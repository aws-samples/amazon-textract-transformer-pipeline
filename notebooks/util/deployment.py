# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Utilities to simplify model/endpoint deployment
"""

# Python Built-Ins:
import errno
import io
from logging import getLogger
import os
import tarfile

# External Dependencies:
import numpy as np
import sagemaker


logger = getLogger("deploy")


def tar_as_inference_code(folder: str, outfile: str = "model.tar.gz") -> str:
    """Package a folder of code (without model artifacts) to run in SageMaker endpoint

    SageMaker framework endpoints expect a .tar.gz archive, and PyTorch/HuggingFace frameworks in
    particular look for a 'code/' folder within this archive with an 'inference.py' entrypoint
    script.

    Given a local folder, this function will produce a .tar.gz file with the folder's contents
    archived under 'code/'. It will warn if the folder does not contain an 'inference.py'.

    Parameters
    ----------
    folder :
        Local folder of code to package
    outfile :
        Local path to write output archive to (default "model.tar.gz")

    Returns
    -------
    outfile :
        (Unchanged) local path to the saved tarball.
    """

    if "inference.py" not in os.listdir(folder):
        logger.warning(
            "Folder '%s' does not contain an 'inference.py' and so won't work as a SM endpoint "
            "bundle unless you make extra configurations on your Model",
            folder,
        )
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    try:  # Remove existing file if present
        os.remove(outfile)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise e
    with tarfile.open(outfile, mode="w:gz") as archive:
        archive.add(
            folder,
            # Name folder explicitly as 'code', as required for modern PyTorchModel versions:
            arcname="code",
            # Exclude hidden files like .ipynb_checkpoints:
            filter=lambda info: None if "/." in info.name else info,
        )
    return outfile


class FileSerializer(sagemaker.serializers.SimpleBaseSerializer):
    """Serializer to simply send contents of a file: predictor.predict(filepath)

    You should set content_type to match your intended files when constructing this serializer. For
    example 'application/pdf', 'image/png', etc.
    """

    EXTENSION_TO_MIME_TYPE = {
        "jpg": "image/jpg",
        "jpeg": "image/jpeg",
        "pdf": "application/pdf",
        "png": "image/png",
    }

    def serialize(self, data: str):
        with open(data, "rb") as f:
            return f.read()

    @classmethod
    def content_type_from_filename(cls, filename: str):
        ext = filename.rpartition(".")[2]
        try:
            return cls.EXTENSION_TO_MIME_TYPE[ext]
        except KeyError as ke:
            pass
        raise ValueError(f"Unknown content type for filename extension '.{ext}'")

    @classmethod
    def from_filename(cls, filename: str, **kwargs):
        return cls(content_type=cls.content_type_from_filename(filename), **kwargs)


class CompressedNumpyDeserializer(sagemaker.deserializers.NumpyDeserializer):
    """Like SageMaker's NumpyDeserializer, but also supports (and defaults to) .npz archive

    While .npy files save an individual array, .npz archives store multiple named variables and
    can be saved with compression to further reduce payload size.
    """

    def __init__(self, dtype=None, accept="application/x-npz", allow_pickle=True):
        super(CompressedNumpyDeserializer, self).__init__(
            dtype=dtype, accept=accept, allow_pickle=allow_pickle
        )

    def deserialize(self, stream, content_type):
        if content_type == "application/x-npz":
            try:
                return np.load(io.BytesIO(stream.read()), allow_pickle=self.allow_pickle)
            finally:
                stream.close()
        else:
            super(CompressedNumpyDeserializer, self).deserialize(stream, content_type)

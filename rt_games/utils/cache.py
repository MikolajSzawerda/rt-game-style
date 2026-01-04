from typing import Any, Optional


class ModelCache:
    """
    Lightweight model cache to avoid reloading heavy nets (VGG, MiDaS, Inception, LPIPS).
    """

    _vgg: Optional[Any] = None
    _midas: Optional[Any] = None
    _inception: Optional[Any] = None
    _art_inception: Optional[Any] = None
    _flow: Optional[Any] = None
    _lpips: Optional[Any] = None

    @classmethod
    def get_vgg(cls, factory, device: str):
        if cls._vgg is None:
            cls._vgg = factory(device)
        return cls._vgg

    @classmethod
    def get_midas(cls, factory, device: str):
        if cls._midas is None:
            cls._midas = factory(device)
        return cls._midas

    @classmethod
    def get_inception(cls, factory, device: str, art: bool = False):
        key = "_art_inception" if art else "_inception"
        if getattr(cls, key) is None:
            setattr(cls, key, factory(device))
        return getattr(cls, key)

    @classmethod
    def get_flow(cls, factory, device: str):
        if cls._flow is None:
            cls._flow = factory(device)
        return cls._flow

    @classmethod
    def get_lpips(cls, factory, device: str):
        if cls._lpips is None:
            cls._lpips = factory(device)
        return cls._lpips

#!/usr/bin/env python3
"""Fixed generic grayscale transduction into translation-invariant local contrast events."""
from __future__ import annotations
from itertools import combinations
from reference_population_v1 import mix64

ORIENTATION_TAG=0xA11CE1
MAGNITUDE_TAG=0xA11CE2
CONTRAST_FLOOR=16
MAGNITUDE_STEP=32


class RawVisualElementsV1:
    """Stateless local contrast transducer with no downstream semantic authority."""
    @staticmethod
    def _matrix(image):
        rows=tuple(tuple(int(v) for v in row) for row in image)
        if len(rows)<2 or any(len(row)<2 for row in rows):raise ValueError('raw_visual:image_shape')
        width=len(rows[0])
        if any(len(row)!=width for row in rows):raise ValueError('raw_visual:image_shape')
        if any(v<0 or v>255 for row in rows for v in row):raise ValueError('raw_visual:pixel_range')
        return rows

    @staticmethod
    def _token(tag:int,*parts:int)->int:
        value=int(tag)
        for part in parts:value=mix64(value^mix64(int(part)+0x9E3779B97F4A7C15))
        return int(value&((1<<63)-1) or 1)

    @classmethod
    def _events(cls,a,b,c,d):
        # x-gradient compares right versus left columns; y-gradient bottom versus top rows.
        dx=(int(b)+int(d))-(int(a)+int(c))
        dy=(int(c)+int(d))-(int(a)+int(b))
        candidates=((1,dx),(2,dy));events=[]
        for axis,diff2 in candidates:
            # diff2 is twice the mean contrast; compare against twice the floor.
            if abs(diff2)<2*CONTRAST_FLOOR:continue
            polarity=1 if diff2>0 else -1
            magnitude=min(7,max(1,(abs(diff2)//2)//MAGNITUDE_STEP+1))
            events.append(cls._token(ORIENTATION_TAG,axis,polarity))
            events.append(cls._token(MAGNITUDE_TAG,magnitude))
        return tuple(sorted(set(events)))

    @classmethod
    def event_windows(cls,image):
        rows=cls._matrix(image);out=[]
        for y in range(len(rows)-1):
            for x in range(len(rows[0])-1):
                events=cls._events(rows[y][x],rows[y][x+1],rows[y+1][x],rows[y+1][x+1])
                if events:out.append(events)
        return tuple(out)

    @classmethod
    def pair_patches(cls,image):
        """All unique within-window event pairs; local grouping is derived, not fixture-supplied."""
        pairs=[]
        for events in cls.event_windows(image):
            for pair in combinations(events,2):pairs.append(tuple(sorted(pair)))
        return tuple(sorted(set(pairs)))

    @classmethod
    def event_vocabulary(cls,image):
        return tuple(sorted({event for window in cls.event_windows(image) for event in window}))

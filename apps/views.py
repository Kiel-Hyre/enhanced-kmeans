from django.shortcuts import render

from rest_framework.views import APIView
from rest_framework import serializers
from rest_framework.response import Response

from .firefly import run

import pandas as pd

import os


def main(request):
    return render(request, template_name="index.html", context={})


class KmeansAPIView(APIView):

    class InputSerializer(serializers.Serializer):
        file = serializers.FileField(required=False)
        n = serializers.FloatField(required=False)
        k = serializers.FloatField(required=False)

    def post(self, request, *args, **kwargs):

        serializer = self.InputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            return Response(run(
                file=serializer.validated_data.get("file", None),
                n=serializer.validated_data.get("n", 0.99),
                k=serializer.validated_data.get("k", 6),
            ))
        except Exception as e:
            return Response({"error": str(e)})

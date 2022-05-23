from django.shortcuts import render

from rest_framework.views import APIView
from rest_framework import serializers
from rest_framework.response import Response

from .firefly import run

import pandas as pd


class KmeansAPIView(APIView):

    class InputSerializer(serializers.Serializer):
        file = serializers.FileField()
        n = serializers.FloatField(required=False)
        k = serializers.FloatField(required=False)

    class OutputSerializer(serializers.Serializer):
        silhouette_score = serializers.FloatField(read_only=True)
        davies_bouldin_score = serializers.FloatField(read_only=True)
        calinski_harabasz_score =serializers.FloatField(read_only=True)

    def post(self, request, *args, **kwargs):

        serializer = self.InputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        orig = pd.read_csv(serializer.validated_data["file"])
        data = orig.copy()

        try:
            # minimal cleaning
            data = data.drop(['REGION', 'PROVINCE', 'CITY'], axis=1)
        except:
            None

        return Response(run(
            data=data,
            orig=orig,
            n=serializer.validated_data.get("n", 0.99),
            k=serializer.validated_data.get("k", 6),
        ))

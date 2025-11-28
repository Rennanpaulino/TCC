package com.seutcc.app.network;

import com.seutcc.app.models.Report;
import com.seutcc.app.models.Trem;
import java.util.List;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.Url;

public interface ApiService {
    // Busca a lista de trens (IoT Service - Porta 8002)
    @GET
    Call<List<Trem>> getTrens(@Url String url);

    // Envia denúncia (Report Service - Porta 8003)
    @POST
    Call<Void> enviarReport(@Url String url, @Body Report report);
}
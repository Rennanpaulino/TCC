package com.seutcc.app.network;

import com.seutcc.app.models.*;
import java.util.List;
import java.util.Map;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;

public interface ApiService {
    // Auth (Porta 8001)
    @POST("login")
    Call<LoginResponse> login(@Body UserLogin login);

    // Auth - Cadastro (NOVO)
    // O Backend retorna uma msg simples: {"msg": "Criado com sucesso"}
    // Podemos usar um Map ou criar uma classe MsgResponse. Vamos usar Map pra ser rápido.
    @POST("register")
    Call<Map<String, String>> register(@Body UserLogin login);

    // IoT (Porta 8002)
    @GET("trens")
    Call<List<Trem>> getTrens();

    // Report (Porta 8003)
    @GET("reports/stats")
    Call<Map<String, Integer>> getStats();
}
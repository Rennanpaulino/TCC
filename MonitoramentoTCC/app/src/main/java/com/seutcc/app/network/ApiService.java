package com.seutcc.app.network;

import com.seutcc.app.models.*;
import java.util.List;
import java.util.Map;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.Path;
public interface ApiService {
    // Auth (Porta 8001)
    @POST("login")
    Call<LoginResponse> login(@Body UserLogin login);

    // Auth - Cadastro (NOVO)
    // O Backend retorna uma msg simples: {"msg": "Criado com sucesso"}
    // Podemos usar um Map ou criar uma classe MsgResponse. Vamos usar Map pra ser rápido.
    @POST("register")
    Call<Map<String, String>> register(@Body UserLogin login);

    @POST("/report") // A rota base já vem do RetrofitClient.getReportService()
    Call<Void> enviarReport(@Body Report report);

    // IoT (Porta 8002)
    @GET("trens")
    Call<List<Trem>> getTrens();

    // Report (Porta 8003)
    @GET("reports/stats")
    Call<Map<String, Integer>> getStats();

    // 1. Busca todas as linhas
    @GET("/linhas")
    Call<List<Linha>> getLinhas();

    // 2. Busca estações de uma linha específica
    @GET("/estacoes/{linhaId}")
    Call<List<Estacao>> getEstacoes(@Path("linhaId") String linhaId);

    // 3. Busca a previsão para a estação selecionada
    @GET("/previsao/{linhaId}/{estacaoId}")
    Call<Previsao> getPrevisao(@Path("linhaId") String linhaId, @Path("estacaoId") String estacaoId);
}
package com.seutcc.app.network;

import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class RetrofitClient {
    // IMPORTANTE: Mude para seu IP

    // --- OPÇÃO A: EMULADOR (Use esta se estiver no PC) ---
    //private static final String BASE_URL = "http://10.0.2.2";
    // --- OPÇÃO B: CELULAR FÍSICO (Use esta se conectar o cabo USB) ---
    private static final String BASE_URL = "http://192.168.15.11";

    private static Retrofit getRetrofit(int port) {
        return new Retrofit.Builder()
                .baseUrl(BASE_URL + ":" + port + "/")
                .addConverterFactory(GsonConverterFactory.create())
                .build();
    }

    public static ApiService getAuthService() {
        return getRetrofit(8001).create(ApiService.class);
    }

    public static ApiService getIoTService() {
        return getRetrofit(8002).create(ApiService.class);
    }

    public static ApiService getReportService() {
        return getRetrofit(8003).create(ApiService.class);
    }
}
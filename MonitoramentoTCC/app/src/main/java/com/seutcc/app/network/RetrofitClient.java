package com.seutcc.app.network;

import okhttp3.OkHttpClient;
import okhttp3.logging.HttpLoggingInterceptor;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class RetrofitClient {

    // GARANTA QUE ESTÁ SEM A BARRA NO FINAL!
    //private static final String BASE_URL = "http://10.0.2.2";
    private static final String BASE_URL = "http://192.168.15.4";
    private static Retrofit getRetrofit(int port) {

        // --- CONFIGURAÇÃO DO LOG ---
        HttpLoggingInterceptor logging = new HttpLoggingInterceptor();
        // Nível BODY mostra o JSON inteiro que vai e volta
        logging.setLevel(HttpLoggingInterceptor.Level.BODY);

        OkHttpClient client = new OkHttpClient.Builder()
                .addInterceptor(logging)
                .build();
        // ---------------------------

        return new Retrofit.Builder()
                .baseUrl(BASE_URL + ":" + port + "/")
                .addConverterFactory(GsonConverterFactory.create())
                .client(client) // Adiciona o cliente com log
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
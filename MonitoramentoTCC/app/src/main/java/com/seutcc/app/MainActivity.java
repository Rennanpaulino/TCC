package com.seutcc.app;

import android.content.DialogInterface;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.widget.EditText;
import android.widget.Toast;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.seutcc.app.adapters.TremAdapter;
import com.seutcc.app.models.Report;
import com.seutcc.app.models.Trem;
import com.seutcc.app.network.ApiService;

import java.util.List;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class MainActivity extends AppCompatActivity {

    private TremAdapter adapter;
    private ApiService apiService;
    private Handler handler = new Handler(Looper.getMainLooper());

    // IP DO SEU PC (Ajuste aqui)
    private String IP_SERVIDOR = "http://192.168.15.9";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 1. Configura a Lista (RecyclerView)
        RecyclerView recyclerView = findViewById(R.id.recyclerTrens);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));

        // Configura o Adapter e o evento de clique
        adapter = new TremAdapter(trem -> mostrarDialogoReport(trem));
        recyclerView.setAdapter(adapter);

        // 2. Configura Retrofit
        // Usamos uma base url genérica pois vamos passar a URL completa nas chamadas
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl(IP_SERVIDOR)
                .addConverterFactory(GsonConverterFactory.create())
                .build();
        apiService = retrofit.create(ApiService.class);

        // 3. Inicia o Loop
        iniciarAtualizacao();
    }

    private void iniciarAtualizacao() {
        handler.post(new Runnable() {
            @Override
            public void run() {
                buscarTrens();
                handler.postDelayed(this, 3000); // Roda a cada 3 segundos
            }
        });
    }

    private void buscarTrens() {
        // Chama a porta 8002 (IoT Service)
        String url = IP_SERVIDOR + ":8002/trens";

        apiService.getTrens(url).enqueue(new Callback<List<Trem>>() {
            @Override
            public void onResponse(Call<List<Trem>> call, Response<List<Trem>> response) {
                if (response.isSuccessful() && response.body() != null) {
                    adapter.atualizarDados(response.body());
                }
            }

            @Override
            public void onFailure(Call<List<Trem>> call, Throwable t) {
                System.out.println("Erro buscando trens: " + t.getMessage());
            }
        });
    }

    private void mostrarDialogoReport(Trem trem) {
        // Cria uma janelinha para digitar o problema
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Reportar Problema: " + trem.getId());

        final EditText input = new EditText(this);
        input.setHint("Descreva o problema (ex: Ar quebrado)");
        builder.setView(input);

        builder.setPositiveButton("Enviar", (dialog, which) -> {
            String problema = input.getText().toString();
            enviarReportBackend(trem.getId(), problema);
        });
        builder.setNegativeButton("Cancelar", (dialog, which) -> dialog.cancel());

        builder.show();
    }

    private void enviarReportBackend(String tremId, String problema) {
        // Chama a porta 8003 (Report Service)
        String url = IP_SERVIDOR + ":8003/report";

        Report report = new Report("UsuarioAndroid", tremId, "Geral", problema);

        apiService.enviarReport(url, report).enqueue(new Callback<Void>() {
            @Override
            public void onResponse(Call<Void> call, Response<Void> response) {
                if (response.isSuccessful()) {
                    Toast.makeText(MainActivity.this, "Report enviado com sucesso!", Toast.LENGTH_SHORT).show();
                } else {
                    Toast.makeText(MainActivity.this, "Erro ao enviar report", Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onFailure(Call<Void> call, Throwable t) {
                Toast.makeText(MainActivity.this, "Falha na conexão", Toast.LENGTH_SHORT).show();
            }
        });
    }
}
package com.seutcc.app;

import android.graphics.Color;
import android.location.Location;
import android.os.Bundle;
import android.os.Handler;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import com.seutcc.app.models.Trem;
import com.seutcc.app.network.RetrofitClient;
import java.util.List;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class DetalhesTremActivity extends AppCompatActivity {

    private String tremId;
    private TextView txtTitulo, txtEta, txtVelocidade, txtStatusGps;
    private ProgressBar progressBar;
    private Handler handler = new Handler();

    // Coordenadas Fixas de Santo Amaro (Destino)
    private final double LAT_DESTINO = -23.65637;
    private final double LON_DESTINO = -46.70956;

    // Distância total da linha fictícia para calcular a barra de progresso (Ex: 10km)
    private final float DISTANCIA_TOTAL_LINHA_METROS = 10000;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detalhes_trem);

        tremId = getIntent().getStringExtra("ID_TREM");

        txtTitulo = findViewById(R.id.txtTituloTrem);
        txtEta = findViewById(R.id.txtEtaGrande);
        txtVelocidade = findViewById(R.id.txtVelocidade);
        txtStatusGps = findViewById(R.id.txtStatusGps);
        progressBar = findViewById(R.id.progressBarTrajeto);

        txtTitulo.setText(tremId);

        iniciarLoop();
    }

    private void iniciarLoop() {
        atualizarDados();
        handler.postDelayed(this::iniciarLoop, 3000); // Roda a cada 3s
    }

    private void atualizarDados() {
        RetrofitClient.getIoTService().getTrens().enqueue(new Callback<List<Trem>>() {
            @Override
            public void onResponse(Call<List<Trem>> call, Response<List<Trem>> response) {
                if (response.isSuccessful() && response.body() != null) {
                    for (Trem t : response.body()) {
                        if (t.getId().equals(tremId)) {
                            atualizarTela(t);
                            break;
                        }
                    }
                }
            }
            @Override
            public void onFailure(Call<List<Trem>> call, Throwable t) {}
        });
    }

    private void atualizarTela(Trem t) {
        // 1. Atualiza Textos
        String etaTexto = String.format("%02d:%02d", t.getEtaMinutos(), t.getEtaSegundos());
        txtEta.setText(etaTexto);
        txtVelocidade.setText(String.format("Velocidade: %.1f km/h", t.getVelocidade()));

        // 2. Status do GPS (Item 2.2 do Backend)
        if (t.isEstimado()) {
            txtStatusGps.setText("⚠️ Sinal GPS Perdido - Usando Estimativa");
            txtStatusGps.setTextColor(Color.parseColor("#FF9800")); // Laranja
        } else {
            txtStatusGps.setText("● Sinal GPS Ativo");
            txtStatusGps.setTextColor(Color.parseColor("#4CAF50")); // Verde
        }

        // 3. Atualiza Barra de Progresso (Matemática simples para visualização)
        float[] results = new float[1];
        Location.distanceBetween(t.getLatitude(), t.getLongitude(), LAT_DESTINO, LON_DESTINO, results);
        float distanciaRestante = results[0];

        // Se faltam 2km de 10km totais, andamos 8km (80%)
        float distanciaPercorrida = DISTANCIA_TOTAL_LINHA_METROS - distanciaRestante;
        int progresso = (int) ((distanciaPercorrida / DISTANCIA_TOTAL_LINHA_METROS) * 100);

        // Limita entre 0 e 100
        if (progresso < 0) progresso = 0;
        if (progresso > 100) progresso = 100;

        progressBar.setProgress(progresso);
    }
}
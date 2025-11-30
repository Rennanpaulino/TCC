package com.seutcc.app;

import android.os.Bundle;
import android.os.Handler;
import android.view.View;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import com.seutcc.app.models.Previsao;
import com.seutcc.app.network.RetrofitClient;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class DetalhesEstacaoActivity extends AppCompatActivity {

    private String linhaId, estacaoId;
    private TextView txtTimer, txtAviso;
    private ProgressBar loading;
    private Handler handler = new Handler();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detalhes_estacao);

        linhaId = getIntent().getStringExtra("LINHA_ID");
        estacaoId = getIntent().getStringExtra("ESTACAO_ID");
        String nome = getIntent().getStringExtra("ESTACAO_NOME");

        ((TextView)findViewById(R.id.txtNomeEstacao)).setText(nome);
        txtTimer = findViewById(R.id.txtTimer);
        txtAviso = findViewById(R.id.txtAvisoGPS);
        loading = findViewById(R.id.progressLoading);

        iniciarLoop();
    }

    private void iniciarLoop() {
        buscarPrevisao();
        handler.postDelayed(this::iniciarLoop, 5000); // Atualiza a cada 5s
    }

    private void buscarPrevisao() {
        RetrofitClient.getIoTService().getPrevisao(linhaId, estacaoId).enqueue(new Callback<Previsao>() {
            @Override
            public void onResponse(Call<Previsao> call, Response<Previsao> response) {
                loading.setVisibility(View.GONE);
                if (response.isSuccessful() && response.body() != null) {
                    Previsao p = response.body();

                    // Formata o Tempo (ex: 04 : 12)
                    String tempo = String.format("%02d : %02d", p.getMinutos(), p.getSegundos());
                    txtTimer.setText(tempo);

                    // Aviso de GPS
                    if (p.isEstimado()) {
                        txtAviso.setVisibility(View.VISIBLE);
                    } else {
                        txtAviso.setVisibility(View.GONE);
                    }
                }
            }
            @Override
            public void onFailure(Call<Previsao> call, Throwable t) {
                loading.setVisibility(View.GONE);
                txtTimer.setText("--");
            }
        });
    }
}
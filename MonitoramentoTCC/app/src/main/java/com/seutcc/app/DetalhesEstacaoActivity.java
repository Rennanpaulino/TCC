package com.seutcc.app;

import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

// Imports para o Alerta e Botão
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import com.google.android.material.floatingactionbutton.ExtendedFloatingActionButton;

// Imports do seu projeto
import com.seutcc.app.models.Previsao;
import com.seutcc.app.models.Report;
import com.seutcc.app.network.RetrofitClient;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class DetalhesEstacaoActivity extends AppCompatActivity {

    // Componentes Visuais
    private TextView txtTimer, txtAviso, txtNomeEstacao;
    private ProgressBar loading;
    private ExtendedFloatingActionButton fabReport;

    // Dados da Estação
    private String linhaId, estacaoId, nomeEstacao;

    // Controlador do Loop de Tempo
    private Handler handler = new Handler(Looper.getMainLooper());

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detalhes_estacao);

        // 1. Recebe os dados da tela anterior
        linhaId = getIntent().getStringExtra("LINHA_ID");
        estacaoId = getIntent().getStringExtra("ESTACAO_ID");
        nomeEstacao = getIntent().getStringExtra("ESTACAO_NOME");

        // 2. Conecta com o XML
        txtNomeEstacao = findViewById(R.id.txtNomeEstacao);
        txtTimer = findViewById(R.id.txtTimer);
        txtAviso = findViewById(R.id.txtAviso); // Lembre que no XML o ID pode ser txtAviso ou txtAvisoGPS
        loading = findViewById(R.id.loading); // Se no XML for progressLoading, ajuste aqui
        fabReport = findViewById(R.id.fabReport);

        // Define o título
        txtNomeEstacao.setText(nomeEstacao);

        // 3. Configura o Botão de Reportar
        fabReport.setOnClickListener(v -> mostrarDialogoReport());

        // 4. Inicia o Loop de Previsão
        iniciarLoop();
    }

    // --- LÓGICA DE PREVISÃO (LOOP) ---

    private void iniciarLoop() {
        handler.post(new Runnable() {
            @Override
            public void run() {
                buscarPrevisao();
                // Agenda a próxima execução para daqui 5 segundos
                handler.postDelayed(this, 5000);
            }
        });
    }

    private void buscarPrevisao() {
        RetrofitClient.getIoTService().getPrevisao(linhaId, estacaoId).enqueue(new Callback<Previsao>() {
            @Override
            public void onResponse(Call<Previsao> call, Response<Previsao> response) {
                loading.setVisibility(View.GONE); // Esconde loading na primeira resposta

                if (response.isSuccessful() && response.body() != null) {
                    Previsao p = response.body();

                    // Verifica mensagens especiais (Ex: "Nenhum trem na linha")
                    if (p.getMensagem() != null && p.getMensagem().contains("Nenhum trem")) {
                        txtTimer.setText("S/ Trem");
                        txtTimer.setTextSize(40); // Diminui fonte se o texto for grande
                        return;
                    }

                    // Formata o Tempo (ex: 04 : 12)
                    String tempo = String.format("%02d : %02d", p.getMinutos(), p.getSegundos());
                    txtTimer.setText(tempo);
                    txtTimer.setTextSize(70); // Volta ao tamanho normal

                    // Aviso de GPS Estimado
                    if (p.isEstimado()) {
                        txtAviso.setVisibility(View.VISIBLE);
                        txtAviso.setText("⚠️ Sinal GPS Instável (Estimado)");
                    } else {
                        txtAviso.setVisibility(View.GONE);
                    }
                }
            }

            @Override
            public void onFailure(Call<Previsao> call, Throwable t) {
                loading.setVisibility(View.GONE);
                txtTimer.setText("-- : --");
                // Não mostramos Toast aqui para não spammar o usuário a cada 5s se a net cair
            }
        });
    }

    // --- LÓGICA DE REPORT (BOTÃO) ---

    private void mostrarDialogoReport() {
        // Lista de problemas para o usuário escolher
        String[] problemas = {
                "Trem muito atrasado",
                "Superlotação Extrema",
                "Segurança / Furto",
                "Sujeira / Limpeza",
                "Ar Condicionado Quebrado",
                "Outros"
        };

        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Qual o problema aqui?");

        builder.setItems(problemas, (dialog, which) -> {
            // "which" é o índice do item clicado (0, 1, 2...)
            String problemaEscolhido = problemas[which];
            enviarReportBackend(problemaEscolhido);
        });

        builder.setNegativeButton("Cancelar", null);
        builder.show();
    }

    private void enviarReportBackend(String problema) {
        // Cria o objeto Report
        // Usuário fixo para TCC, mas poderia pegar do SessionManager
        Report report = new Report("PassageiroApp", linhaId, estacaoId, problema);

        // Chama o serviço de Report (Porta 8003)
        RetrofitClient.getReportService().enviarReport(report).enqueue(new Callback<Void>() {
            @Override
            public void onResponse(Call<Void> call, Response<Void> response) {
                if (response.isSuccessful()) {
                    Toast.makeText(DetalhesEstacaoActivity.this, "Ocorrência registrada! Obrigado.", Toast.LENGTH_LONG).show();
                } else {
                    Toast.makeText(DetalhesEstacaoActivity.this, "Erro ao registrar: " + response.code(), Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onFailure(Call<Void> call, Throwable t) {
                Toast.makeText(DetalhesEstacaoActivity.this, "Falha ao enviar report.", Toast.LENGTH_SHORT).show();
            }
        });
    }

    // Boa prática: Parar o loop se o usuário sair da tela para economizar bateria
    @Override
    protected void onDestroy() {
        super.onDestroy();
        handler.removeCallbacksAndMessages(null); // Mata o loop
    }
}